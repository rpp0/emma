import socket
import logging
import sys
import pickle
import struct
from emma.io.traceset import TraceSet

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


class EMCapOnlineClient:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.trace_set_count = 0

    def connect(self, ip, port=3885):
        self.socket.connect((ip, port))

    def send(self, np_signals, np_plaintexts, np_ciphertexts, np_keys, np_masks):
        ts = TraceSet(name="online %d" % self.trace_set_count, traces=np_signals, plaintexts=np_plaintexts,
                      ciphertexts=np_ciphertexts, keys=np_keys, masks=np_masks)
        logger.info("Pickling")
        ts_p = pickle.dumps(ts)
        logger.info("Size is %d" % len(ts_p))
        stream_payload = ts_p
        stream_payload_len = len(stream_payload)
        logger.info("Streaming trace set of %d bytes to server" % stream_payload_len)
        stream_hdr = struct.pack(">BI", 0, stream_payload_len)
        self.socket.send(stream_hdr + stream_payload)
        self.trace_set_count += 1
