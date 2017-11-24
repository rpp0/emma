#!/usr/bin/python2

from gnuradio import blocks
from gnuradio import eng_notation
from gnuradio import gr
from gnuradio import uhd
from gnuradio.eng_option import eng_option
from gnuradio.filter import firdes
from time import sleep
from threading import Thread
from datetime import datetime
from sigmf.sigmffile import SigMFFile
from dsp import butter_filter
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import socket
import os
import signal
import logging
import struct
import binascii
import osmosdr
import argparse

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def handler(signum, frame):
    print("Got CTRL+C")
    exit(0)

signal.signal(signal.SIGINT, handler)

def binary_to_hex(binary):
    result = []
    for elem in binary:
        result.append("{:0>2}".format(binascii.hexlify(elem)))
    return ' '.join(result)

class CtrlPacketType:
    SIGNAL_START = 0
    SIGNAL_END = 1

# SDR capture device
class SDR(gr.top_block):
    def __init__(self, hw="usrp", samp_rate=100000, freq=3.2e9, gain=0):
        gr.enable_realtime_scheduling()
        gr.top_block.__init__(self, "SDR capture device")

        ##################################################
        # Variables
        ##################################################
        self.hw = hw
        self.samp_rate = samp_rate
        self.freq = freq
        self.gain = gain
        logger.info("%s: samp_rate=%d, freq=%f, gain=%d" % (hw, samp_rate, freq, gain))

        ##################################################
        # Blocks
        ##################################################
        if hw == "usrp":
            self.sdr_source = uhd.usrp_source(  # Set recv_frame_size to 65536 and num_recv_frames to 1024
               ",".join(("", "recv_frame_size=4096", "num_recv_frames=1024")),
               #",".join(("", "")),
               uhd.stream_args(
               cpu_format="fc32",
               channels=range(1),
               ),
            )
            self.sdr_source.set_samp_rate(samp_rate)
            self.sdr_source.set_center_freq(freq, 0)
            self.sdr_source.set_gain(gain, 0)
            self.sdr_source.set_min_output_buffer(16*1024*1024)  # 16 MB output buffer
        else:
            self.sdr_source = osmosdr.source(args="numchan=" + str(1) + " " + '' )
            self.sdr_source.set_sample_rate(samp_rate)
            self.sdr_source.set_center_freq(freq, 0)
            self.sdr_source.set_freq_corr(0, 0)
            self.sdr_source.set_dc_offset_mode(0, 0)
            self.sdr_source.set_iq_balance_mode(0, 0)
            self.sdr_source.set_gain_mode(False, 0)
            self.sdr_source.set_gain(gain, 0)
            self.sdr_source.set_if_gain(20, 0)
            self.sdr_source.set_bb_gain(20, 0)
            self.sdr_source.set_antenna('', 0)
            self.sdr_source.set_bandwidth(0, 0)

        self.udp_sink = blocks.udp_sink(8, "127.0.0.1", 3884, payload_size=1472, eof=True)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.sdr_source, 0), (self.udp_sink, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        if hw == "usrp":
            self.sdr_source.set_samp_rate(self.samp_rate)
        else:
            self.sdr_source.set_sample_rate(self.sample_rate)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.sdr_source.set_center_freq(self.freq, 0)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.sdr_source.set_gain(self.gain, 0)

class SocketWrapper(Thread):
    def __init__(self, s, address, cb_pkt):
        Thread.__init__(self)
        self.setDaemon(True)
        self.socket = s
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        logger.debug("Binding to %s" % str(address))
        self.socket.bind(address)
        self.cb_pkt = cb_pkt
        self.data = b""

    def _parse(self, client_socket, client_address):
        bytes_parsed = self.cb_pkt(client_socket, client_address, self.data)
        self.data = self.data[bytes_parsed:]

    def recv_stream(self):
        self.socket.listen(1)

        # Only accept one connection
        client_socket, client_address = self.socket.accept()
        logger.debug("Client %s connected" % str(client_address))
        try:
            streaming = True
            while streaming:
                # Add chunk to data
                chunk = client_socket.recv(1024)
                if chunk:
                    self.data += chunk
                else:
                    streaming = False
                    logger.debug("Received null, stopping soon!")

                # Parse existing data
                self._parse(client_socket, client_address)
        finally:
            client_socket.close()

    def recv_dgram(self):
        receiving = True
        while receiving:
            chunk, client_address = self.socket.recvfrom(1472)
            if chunk:
                self.data += chunk
            else:
                receiving = False
                logger.debug("Received null packet, stopping soon!")

            self._parse(None, client_address)

    def run(self):
        '''
        Choose a suitable receiver depending on the type of socket
        '''
        if self.socket.type == socket.SOCK_DGRAM:
            self.recv_dgram()
        elif self.socket.type == socket.SOCK_STREAM:
            self.recv_stream()
        else:
            logger.error("Unrecognized socket type %s" % self.socket.type)

# EMCap class: wait for signal and start capturing using a SDR
class EMCap():
    def __init__(self, cap_kwargs={}):
        unix_domain_socket = '/tmp/emma.socket'

        self.clear_domain_socket(unix_domain_socket)
        self.data_socket = SocketWrapper(socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM), ('127.0.0.1', 3884), self.cb_data)
        self.ctrl_socket = SocketWrapper(socket.socket(family=socket.AF_UNIX, type=socket.SOCK_STREAM), unix_domain_socket, self.cb_ctrl)

        self.sdr = SDR(**cap_kwargs)
        self.cap_kwargs = cap_kwargs
        self.store = False
        self.stored_plaintext = []
        self.stored_data = []
        self.trace_set = []
        self.plaintexts = []

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
        logger.debug("Control packet: " + binary_to_hex(data))
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
                client_socket.sendall("k")
                return payload_len + 5

    def process_ctrl_packet(self, pkt_type, payload):
        if pkt_type == CtrlPacketType.SIGNAL_START:
            logger.debug(payload)
            self.stored_plaintext = [ord(c) for c in payload]
            self.sdr.start()

            # Spinlock until data
            timeout = 3
            current_time = 0.0
            while len(self.stored_data) == 0:
                sleep(0.001)
                current_time += 0.001
                if current_time >= timeout:
                    logger.warning("Timeout while waiting for data. Did the SDR crash? Reinstantiating...")
                    del self.sdr
                    self.data_socket.socket.close()
                    self.data_socket = SocketWrapper(socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM), ('127.0.0.1', 3884), self.cb_data)
                    self.data_socket.start()
                    self.sdr = SDR(**self.cap_kwargs)
                    self.process_ctrl_packet(pkt_type, payload)
        elif pkt_type == CtrlPacketType.SIGNAL_END:
            self.sdr.stop()
            self.sdr.wait()

            # Successful capture (no errors or timeouts)
            if len(self.stored_data) > 0:
                # Data to file
                np_data = np.fromstring(b"".join(self.stored_data), dtype=np.complex64)
                self.trace_set.append(np.abs(np_data))
                self.plaintexts.append(self.stored_plaintext)
                #np.save(...)

                # Write metadata to sigmf file
                if len(self.trace_set) >= 256:
                    assert(len(self.trace_set) == len(self.plaintexts))
                    # if sigmf
                    #with open(test_meta_path, 'w') as f:
                    #    test_sigmf = SigMFFile(data_file=test_data_path, global_info=copy.deepcopy(self.global_meta))
                    #    test_sigmf.add_capture(0, metadata=capture_meta)
                    #    test_sigmf.dump(f, pretty=True)
                    # elif chipwhisperer:
                    logger.info("Dumping %d traces to file" % len(self.trace_set))
                    np_trace_set = np.array(self.trace_set)
                    np_plaintexts = np.array(self.plaintexts, dtype=np.uint8)
                    filename = str(datetime.utcnow()).replace(" ","_").replace(".","_")
                    np.save("/home/pieter/projects/em/traces/%s_traces.npy" % filename, np_trace_set)  # TODO abstract this in trace_set class
                    np.save("/home/pieter/projects/em/traces/%s_textin.npy" % filename, np_plaintexts)
                    self.trace_set = []
                    self.plaintexts = []

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

# Test function
def main():
    parser = argparse.ArgumentParser(description='EMCAP')
    parser.add_argument('hw', type=str, choices=['usrp', 'hackrf'], help='SDR capture hardware')
    parser.add_argument('--sample-rate', type=int, default=4000000, help='Sample rate')
    parser.add_argument('--frequency', type=float, default=3.2e9, help='Capture frequency')
    parser.add_argument('--gain', type=int, default=10, help='RX gain')
    args, unknown = parser.parse_known_args()
    e = EMCap(cap_kwargs={'hw': args.hw, 'samp_rate': args.sample_rate, 'freq': args.frequency, 'gain': args.gain})
    e.capture()

if __name__ == '__main__':
    main()
