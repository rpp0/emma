#!/usr/bin/env python3

"""
Tool for automated capturing of EM traces. EMcap can send commands to the target device for starting and stopping
operations using a simple communication protocol over either a serial connection or over TCP.
"""

import numpy as np
import sys
import socket
import os
import signal
import logging
import struct
import binascii
import osmosdr
import argparse
import serial
import subprocess
from gnuradio import blocks
from gnuradio import gr
from gnuradio import uhd
from time import sleep
from datetime import datetime
from emma.utils.socketwrapper import SocketWrapper
from scipy.signal import hilbert
from scipy import fftpack
from emma.emcap.online_client import EMCapOnlineClient
from collections import defaultdict
from emma.emcap.sdr import SDR
from emma.emcap.types import *
from emma.emcap.ttywrapper import TTYWrapper
from emma.utils.utils import binary_to_hex

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

hilbert3 = lambda x: hilbert(x, fftpack.next_fast_len(len(x)))[:len(x)]


def handler(signum, frame):
    print("Got CTRL+C")
    exit(0)


signal.signal(signal.SIGINT, handler)


def reset_usrp():
    print("Resetting USRP")
    p = subprocess.Popen(["/usr/lib/uhd/utils/b2xx_fx3_utils", "--reset-device"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(p.communicate())


# EMCap class: wait for signal and start capturing using a SDR
class EMCap:
    def __init__(self, args):
        # Determine ctrl socket type
        self.ctrl_socket_type = None
        if args.ctrl == 'serial':
            self.ctrl_socket_type = CtrlType.SERIAL
        elif args.ctrl == 'udp':
            self.ctrl_socket_type = CtrlType.UDP

        # Set up data socket
        self.data_socket = SocketWrapper(socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM), ('127.0.0.1', 3884), self.cb_data)
        self.online = args.online

        # Set up sockets
        if self.ctrl_socket_type == CtrlType.DOMAIN:
            unix_domain_socket = '/tmp/emma.socket'
            self.clear_domain_socket(unix_domain_socket)
            self.ctrl_socket = SocketWrapper(socket.socket(family=socket.AF_UNIX, type=socket.SOCK_STREAM), unix_domain_socket, self.cb_ctrl)
        elif self.ctrl_socket_type == CtrlType.UDP:
            self.ctrl_socket = SocketWrapper(socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM), ('10.0.0.1', 3884), self.cb_ctrl)
        elif self.ctrl_socket_type == CtrlType.SERIAL:
            self.ctrl_socket = TTYWrapper("/dev/ttyUSB0", self.cb_ctrl)
        else:
            logger.error("Unknown ctrl_socket_type")
            exit(1)

        if self.online is not None:
            try:
                self.emma_client = EMCapOnlineClient()
                self.emma_client.connect(self.online, 3885)
            except Exception as e:
                print(e)
                exit(1)

        self.sdr_args = {'hw': args.hw, 'samp_rate': args.sample_rate, 'freq': args.frequency, 'gain': args.gain, 'ds_mode': args.ds_mode, 'agc': args.agc}
        self.sdr = SDR(**self.sdr_args)
        self.store = False
        self.stored_plaintext = []
        self.stored_key = []
        self.stored_data = []
        self.trace_set = []
        self.plaintexts = []
        self.keys = []
        self.preprocessed = []
        self.preprocessed_keys = []
        self.preprocessed_plaintexts = []
        self.limit_counter = 0
        self.limit = args.limit
        self.compress = args.compress
        self.args = args

        if self.sdr.hw == 'usrp':
            self.wait_num_chunks = 0
        else:
            self.wait_num_chunks = 50  # Bug in rtl-sdr?

        self.global_meta = {
            "core:datatype": "cf32_le",
            "core:version": "0.0.1",
            "core:license": "CC0",
            "core:hw": self.sdr.hw,
            "core:sample_rate": self.sdr.samp_rate,
            "core:author": "Pieter Robyns"
        }

        self.capture_meta = {
            "core:sample_start": 0,
            "core:frequency": self.sdr.freq,
            "core:datetime": str(datetime.utcnow()),
        }

    def clear_domain_socket(self, address):
        try:
            os.unlink(address)
        except OSError:
            if os.path.exists(address):
                raise

    def cb_timeout(self):
        logger.warning("Timeout on capture, skipping...")
        self.sdr.stop()

    def cb_data(self, client_socket, client_address, data):
        self.stored_data.append(data)
        return len(data)

    def cb_ctrl(self, client_socket, client_address, data):
        logger.log(logging.NOTSET, "Control packet: %s" % binary_to_hex(data))
        if len(data) < 5:
            # Not enough for TLV
            return 0
        else:
            pkt_type, payload_len = struct.unpack(">BI", data[0:5])
            payload = data[5:]
            if len(payload) < payload_len:
                return 0  # Not enough for payload
            else:
                self.process_ctrl_packet(pkt_type, payload)
                # Send ack
                if self.ctrl_socket_type == CtrlType.SERIAL:
                    client_socket.write(b"k")
                else:
                    client_socket.sendall("k")
                return payload_len + 5

    def parse_ies(self, payload):
        while len(payload) >= 5:
            # Extract IE header
            ie_type, ie_len = struct.unpack(">BI", payload[0:5])
            payload = payload[5:]

            # Extract IE data
            ie = payload[0:ie_len]
            payload = payload[ie_len:]
            logger.debug("IE type %d of len %d: %s" % (ie_type, ie_len, binary_to_hex(ie)))

            # Determine what to do with IE
            if ie_type == InformationElementType.PLAINTEXT:
                self.stored_plaintext = [byte_value for byte_value in ie]
            elif ie_type == InformationElementType.KEY:
                self.stored_key = [byte_value for byte_value in ie]
            else:
                logger.warning("Unknown IE type: %d" % ie_type)

    def preprocess(self, trace_set, plaintexts, keys):
        all_traces = []
        key_set = defaultdict(set)
        pt_set = defaultdict(set)

        for i, trace in enumerate(trace_set):
            if len(trace) < 16384:
                print("--skip")
                continue
            trace = trace[8192:16384]
            trace = np.square(np.abs(np.fft.fft(trace)))
            all_traces.append(trace)

            # Check keys and plaintexts
            num_key_bytes = keys.shape[1]
            for j in range(0, num_key_bytes):
                key_set[j].add(keys[i][j])
                pt_set[j].add(plaintexts[i][j])

                if len(key_set[j]) != 1 or len(pt_set[j]) != 1:
                    print("Keys or plaintexts not equal at index %d" % j)
                    print(key_set)
                    print(pt_set)
                    exit(1)

        all_traces = np.array(all_traces)
        self.preprocessed.append(np.mean(all_traces, axis=0))
        self.preprocessed_plaintexts.append(plaintexts[0])
        self.preprocessed_keys.append(keys[0])

    def save(self, trace_set, plaintexts, keys, ciphertexts=None):
        filename = str(datetime.utcnow()).replace(" ", "_").replace(".", "_").replace(":", "-")
        output_dir = self.args.output_dir

        if self.args.preprocess:
            self.preprocess(trace_set, plaintexts, keys)
            if len(self.preprocessed) >= self.args.traces_per_set:
                logger.info("Dumping %d preprocessed traces to file" % len(self.preprocessed))
                np.save(os.path.join(output_dir, "%s_traces.npy" % filename), np.array(self.preprocessed))
                np.save(os.path.join(output_dir, "%s_textin.npy" % filename), np.array(self.preprocessed_plaintexts))
                np.save(os.path.join(output_dir, "%s_knownkey.npy" % filename), np.array(self.preprocessed_keys))
                self.preprocessed = []
                self.preprocessed_plaintexts = []
                self.preprocessed_keys = []
        else:
            logger.info("Dumping %d traces to file" % len(self.trace_set))
            np.save(os.path.join(output_dir, "%s_traces.npy" % filename), trace_set)
            np.save(os.path.join(output_dir, "%s_textin.npy" % filename), plaintexts)
            np.save(os.path.join(output_dir, "%s_knownkey.npy" % filename), keys)
            if self.compress:
                logger.info("Calling emcap-compress...")
                subprocess.call(['/usr/bin/python', 'emcap-compress.py', os.path.join(output_dir, "%s_traces.npy" % filename)])

    def process_ctrl_packet(self, pkt_type, payload):
        if pkt_type == CtrlPacketType.SIGNAL_START:
            logger.debug("Starting for payload: %s" % binary_to_hex(payload))
            self.parse_ies(payload)
            self.sdr.start()

            # Spinlock until data
            timeout = 3
            current_time = 0.0
            while len(self.stored_data) <= self.wait_num_chunks:
                sleep(0.0001)
                current_time += 0.0001
                if current_time >= timeout:
                    logger.warning("Timeout while waiting for data. Did the SDR crash? Reinstantiating...")
                    del self.sdr
                    self.data_socket.socket.close()
                    self.data_socket = SocketWrapper(socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM), ('127.0.0.1', 3884), self.cb_data)
                    self.data_socket.start()
                    self.sdr = SDR(**self.sdr_args)
                    self.process_ctrl_packet(pkt_type, payload)
        elif pkt_type == CtrlPacketType.SIGNAL_END:
            # self.sdr.sdr_source.stop()
            self.sdr.stop()
            self.sdr.wait()

            logger.debug("Stopped after receiving %d chunks" % len(self.stored_data))
            # sleep(0.5)
            # logger.debug("After sleep we have %d chunks" % len(self.stored_data))

            # Successful capture (no errors or timeouts)
            if len(self.stored_data) > 0:  # We have more than 1 chunk
                # Data to file
                np_data = np.fromstring(b"".join(self.stored_data), dtype=np.complex64)
                self.trace_set.append(np_data)
                self.plaintexts.append(self.stored_plaintext)
                self.keys.append(self.stored_key)

                if len(self.trace_set) >= self.args.traces_per_set:
                    assert(len(self.trace_set) == len(self.plaintexts))
                    assert(len(self.trace_set) == len(self.keys))

                    np_trace_set = np.array(self.trace_set)
                    np_plaintexts = np.array(self.plaintexts, dtype=np.uint8)
                    np_keys = np.array(self.keys, dtype=np.uint8)

                    if self.online is not None:  # Stream online
                        self.emma_client.send(np_trace_set, np_plaintexts, None, np_keys, None)
                    else:  # Save to disk
                        if not self.args.dry:
                            # Write metadata to sigmf file
                            # if sigmf
                            #with open(test_meta_path, 'w') as f:
                            #    test_sigmf = SigMFFile(data_file=test_data_path, global_info=copy.deepcopy(self.global_meta))
                            #    test_sigmf.add_capture(0, metadata=capture_meta)
                            #    test_sigmf.dump(f, pretty=True)
                            # elif chipwhisperer:
                            self.save(np_trace_set, np_plaintexts, np_keys)
                        else:
                            print("Dry run! Not saving.")

                        self.limit_counter += len(self.trace_set)
                        if self.limit_counter >= self.limit:
                            print("Done")
                            exit(0)

                    # Clear results
                    self.trace_set = []
                    self.plaintexts = []
                    self.keys = []

                # Clear
                self.stored_data = []
                self.stored_plaintext = []

    def capture(self, to_skip=0, timeout=1.0):
        # Start listening for signals
        self.data_socket.start()
        self.ctrl_socket.start()

        # Wait until supplicant signals end of acquisition
        while self.ctrl_socket.is_alive():
            self.ctrl_socket.join(timeout=1.0)

        logging.info("Supplicant disconnected on control channel. Stopping...")


def main():
    parser = argparse.ArgumentParser(description='EMCAP')
    parser.add_argument('hw', type=str, choices=['usrp', 'hackrf', 'rtlsdr'], help='SDR capture hardware')
    parser.add_argument('ctrl', type=str, choices=['serial', 'udp'], help='Controller type')
    parser.add_argument('--sample-rate', type=int, default=4000000, help='Sample rate')
    parser.add_argument('--frequency', type=float, default=64e6, help='Capture frequency')
    parser.add_argument('--gain', type=float, default=50, help='RX gain')
    parser.add_argument('--traces-per-set', type=int, default=256, help='Number of traces per set')
    parser.add_argument('--limit', type=int, default=256*400, help='Limit number of traces')
    parser.add_argument('--output-dir', dest="output_dir", type=str, default="/tmp/", help='Output directory to store samples')
    parser.add_argument('--online', type=str, default=None, help='Stream samples to remote EMMA instance at <IP address> for online processing.')
    parser.add_argument('--dry', default=False, action='store_true', help='Do not save to disk.')
    parser.add_argument('--ds-mode', default=False, action='store_true', help='Direct sampling mode.')
    parser.add_argument('--agc', default=False, action='store_true', help='Automatic Gain Control.')
    parser.add_argument('--compress', default=False, action='store_true', help='Compress using emcap-compress.')
    parser.add_argument('--preprocess', default=False, action='store_true', help='Preprocess before storing')  # TODO integrate into emcap.py
    args, unknown = parser.parse_known_args()

    e = EMCap(args)
    e.capture()


if __name__ == '__main__':
    main()
