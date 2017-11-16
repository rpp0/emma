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
import numpy as np
import time
import sys
import socket
import os
import signal
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

def handler(signum, frame):
    print("Got CTRL+C")
    exit(0)

signal.signal(signal.SIGINT, handler)

# USRP capture device
class USRP(gr.top_block):
    def __init__(self, samp_rate=100000, freq=1.6e9, gain=0):
        gr.top_block.__init__(self, "USRP capture device")

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate
        self.freq = freq
        self.gain = gain

        ##################################################
        # Blocks
        ##################################################
        self.uhd_usrp_source = uhd.usrp_source(  # Set recv_frame_size to 65536 and num_recv_frames to 1024
        	",".join(("", "recv_frame_size=4096", "num_recv_frames=1024")),
            #",".join(("", "")),
        	uhd.stream_args(
        		cpu_format="fc32",
        		channels=range(1),
        	),
        )
        self.uhd_usrp_source.set_samp_rate(samp_rate)
        self.uhd_usrp_source.set_center_freq(freq, 0)
        self.uhd_usrp_source.set_gain(gain, 0)
        self.uhd_usrp_source.set_min_output_buffer(16*1024*1024)  # 16 MB output buffer
        self.udp_sink = blocks.udp_sink(8, "127.0.0.1", 3884, payload_size=1472, eof=True)

        ##################################################
        # Connections
        ##################################################
        self.connect((self.uhd_usrp_source, 0), (self.udp_sink, 0))

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.uhd_usrp_source.set_samp_rate(self.samp_rate)

    def get_freq(self):
        return self.freq

    def set_freq(self, freq):
        self.freq = freq
        self.uhd_usrp_source.set_center_freq(self.freq, 0)

    def get_gain(self):
        return self.gain

    def set_gain(self, gain):
        self.gain = gain
        self.uhd_usrp_source.set_gain(self.gain, 0)

class SocketWrapper(Thread):
    def __init__(self, s, address, cb_pkt):
        Thread.__init__(self)
        self.setDaemon(True)
        self.socket = s
        logger.info("Binding to %s" % str(address))
        self.socket.bind(address)
        self.cb_pkt = cb_pkt
        self.data = b""

    def _parse(self, client_address):
        bytes_parsed = self.cb_pkt(client_address, self.data)
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
                    logger.info("Received null, stopping soon!")

                # Parse existing data
                self._parse(client_address)
        finally:
            client_socket.close()

    def recv_dgram(self):
        receiving = True
        while receiving:
            chunk, client_address = self.socket.recvfrom(1024)
            if chunk:
                self.data += chunk
            else:
                receiving = False
                logger.info("Received null packet, stopping soon!")

            self._parse(client_address)

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
        self.data_socket = SocketWrapper(socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM), ('127.0.0.1', 3883), self.cb_data)
        self.ctrl_socket = SocketWrapper(socket.socket(family=socket.AF_UNIX, type=socket.SOCK_STREAM), unix_domain_socket, self.cb_ctrl)

        self.usrp = USRP(**cap_kwargs)
        self.store = False

        self.global_meta = {
            "core:datatype": "cf32_le",
            "core:version": "0.0.1",
            "core:license": "CC0",
            "core:hw": "USRP",
            "core:sample_rate": self.usrp.samp_rate,
            "core:author": "Pieter Robyns"
        }

        self.capture_meta = {
            "core:sample_start": 0,
            "core:frequency": self.usrp.freq,
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
        self.usrp.stop()

    def cb_data(self, client_address, data):
        if self.store:
            # We are in a region of interest, so write data to file
            #np_data = np.array(data, dtype=np.float32)
            #np.abs(np_data)
            #np.save(...)

            # Write metadata to sigmf file
            with open(test_meta_path, 'w') as f:
                test_sigmf = SigMFFile(data_file=test_data_path, global_info=copy.deepcopy(self.global_meta))
                test_sigmf.add_capture(0, metadata=capture_meta)
                test_sigmf.dump(f, pretty=True)
        else:
            # Unimportant data
            print("Data is now: " + data)

        return len(data)

    def cb_ctrl(self, client_address, data):
        print("Got ctrl: " + data)
        return len(data)

    def capture(self, to_skip=0, timeout=1.0):
        # Start capturing and listening for data
        self.usrp.start()
        self.data_socket.start()
        self.ctrl_socket.start()

        # Wait until supplicant signals end of acquisition
        while self.ctrl_socket.is_alive():
            self.ctrl_socket.join(timeout=1.0)
        logging.info("Supplicant disconnected on control channel. Stopping...")

        # Stop capturing data with the SDR
        self.usrp.stop()

# Test function
def main():
    e = EMCap()
    e.capture()

if __name__ == '__main__':
    main()
