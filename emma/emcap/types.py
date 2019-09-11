class CtrlPacketType:
    SIGNAL_START = 0
    SIGNAL_END = 1


class InformationElementType:
    PLAINTEXT = 0
    KEY = 1
    CIPHERTEXT = 2
    MASK = 3


class CtrlType:
    DOMAIN = 0
    UDP = 1
    SERIAL = 2
