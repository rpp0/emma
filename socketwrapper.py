from threading import Thread

import socket
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
                chunk = client_socket.recv(2**18)
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
