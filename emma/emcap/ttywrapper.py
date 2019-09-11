import logging
import serial
from threading import Thread

logger = logging.getLogger(__name__)


class TTYWrapper(Thread):
    def __init__(self, port, cb_pkt):
        Thread.__init__(self)
        self.setDaemon(True)
        self.port = port
        logger.debug("Connecting to %s" % str(port))
        self.s = serial.Serial(port, 115200)
        self.cb_pkt = cb_pkt
        self.data = b""

    def _parse(self, client_socket, client_address):
        bytes_parsed = self.cb_pkt(client_socket, client_address, self.data)
        self.data = self.data[bytes_parsed:]

    def recv(self):
        receiving = True
        while receiving:
            if self.s.is_open:
                chunk = self.s.read(1)
                self.data += chunk
            else:
                receiving = False
                logger.debug("Serial connection is closed, stopping soon!")

            self._parse(self.s, None)

    def run(self):
        self.recv()
