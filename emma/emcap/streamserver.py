import configparser
import socket
import struct
import pickle

from queue import Queue
from emma.utils.socketwrapper import SocketWrapper
from emma.utils.utils import get_ip_address
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)


class StreamServer():
    def __init__(self, conf):
        self.conf = conf
        self.queue = Queue()

        if self.conf.online:
            settings = configparser.RawConfigParser()
            settings.read('settings.conf')
            interface = settings.get("Datasets", "stream_interface")
            ip_address = get_ip_address(interface)
            addr_tuple = (ip_address, 3885)

            self.server = SocketWrapper(socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM), addr_tuple, self._cb_server)
            self.server.start()

            print("Listening for sample streams at %s" % str(addr_tuple))

    def _cb_server(self, client_socket, client_address, data):
        if len(data) < 5:
            # Not enough for TLV
            return 0
        else:
            pkt_type, payload_len = struct.unpack(">BI", data[0:5])
            payload = data[5:]
            if len(payload) < payload_len:
                return 0  # Not enough for payload
            else:
                # Depickle and add to queue
                # TODO: Check for correctness. EMcap is Python2 (because it needs to)
                # use GNU Radio. Therefore the pickling format is different, which
                # we need to make sure doesn't cause any differences.
                trace_set = pickle.loads(payload, encoding='latin1', fix_imports=True)
                logger.debug("Stream: got %d traces" % len(trace_set.traces))

                self.queue.put(trace_set)
                return payload_len + 5
