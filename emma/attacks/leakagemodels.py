import numpy as np
import struct
from emma.utils.utils import EMMAException, int_to_one_hot
from emma.attacks.lut import hw, sbox, hw32


class LeakageModelType:
    """
    Class that defines all possible types of leakages. Leakage model classes must have an attribute 'leakage_type' with
    one of the values defined in this class.
    """
    KEY = 'key'
    KEY_OH = 'key_oh'
    KEY_HW = 'key_hw'
    KEY_HW_OH = 'key_hw_oh'
    KEY_BITS = 'key_bits'
    KEY_BIT = 'key_bit'
    HAMMING_WEIGHT_SBOX = 'hamming_weight_sbox'
    HAMMING_WEIGHT_SBOX_OH = 'hamming_weight_sbox_oh'
    HAMMING_WEIGHT_MASKED_SBOX = 'hamming_weight_masked_sbox'
    SBOX = 'sbox'
    SBOX_OH = 'sbox_oh'
    AES_MULTI_TEST = 'aes_test'
    AES_MULTI = 'aes_multi'
    AES_BITS_EX = 'aes_bits_ex'
    HMAC_BITS = 'hmac_bits'
    HMAC_HW = 'hmac_hw'
    HMAC_HW_OH = 'hmac_hw_oh'
    HMAC = 'hmac'

    @classmethod
    def choices(cls):
        """
        Get all possible LeakageModelTypes in list form
        :return:
        """
        c = []
        for k, v in cls.__dict__.items():
            if k[:2] != '__' and type(v) is str:
                c.append(v)
        return c


class LeakageModelMeta(type):
    """
    Metaclass used for checking whether the child class contains valid leakage_type and num_outputs attributes.
    """
    class BadLeakageModelClassException(EMMAException):
        pass

    class InvalidModelTypeException(EMMAException):
        pass

    def __new__(mcs, name, bases, class_dict):
        if bases != (object,):  # Do not validate LeakageModel class
            if 'leakage_type' not in class_dict:
                raise LeakageModelMeta.BadLeakageModelClassException
            if class_dict['leakage_type'] not in LeakageModelType.choices():
                raise LeakageModelMeta.InvalidModelTypeException
        return type.__new__(mcs, name, bases, class_dict)


class LeakageModel(object, metaclass=LeakageModelMeta):
    """
    Leakage model base class.
    """
    class UnknownLeakageModelException(EMMAException):
        pass

    def __new__(cls, conf):
        """
        Called when instantiating a LeakageModel object. Returns an instance of the appropriate class depending on the
        leakage_type parameter.
        :param conf:
        :return:
        """
        for subclass in cls._get_subclasses():
            if subclass.leakage_type == conf.leakage_model:
                return object.__new__(subclass)  # Avoid recursion by calling object.__new__ instead of cls.__new__
        raise LeakageModel.UnknownLeakageModelException

    def __init__(self, conf):
        self.conf = conf
        self.num_subkeys = conf.key_high - conf.key_low  # Number of subkeys
        self.subkey_size = 1  # Size of a subkey in bytes
        self.num_outputs = (self.num_subkeys, 1)  # Leakage model output label size. Dimensions: [num_subkeys, leakage_outputs_per_subkey (default 1)]

    @classmethod
    def _get_subclasses(cls):
        for subclass in cls.__subclasses__():
            if cls is not object:
                for subsubclass in subclass._get_subclasses():  # Also yield children of children
                    yield subsubclass
            yield subclass

    @classmethod
    def get_num_outputs(cls, conf):
        instance = cls(conf)
        return int(np.product(instance.num_outputs))

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        raise NotImplementedError

    def get_trace_set_leakages(self, trace_set, key_hypothesis=None):
        """
        Return numpy array containing the leakage model for [trace_index, key_byte_index]
        :param trace_set:
        :return:
        """
        values = np.zeros((len(trace_set.traces), *self.num_outputs), dtype=float)  # Dimensions: [num_traces, num_outputs]

        for i in range(len(trace_set.traces)):
            for j in range(self.num_outputs[0]):  # num_subkeys
                values[i, j] = self.get_trace_leakages(trace_set.traces[i], j + self.conf.key_low, key_hypothesis=key_hypothesis)

        return values.reshape((len(trace_set.traces), -1))


class HammingWeightSboxLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HAMMING_WEIGHT_SBOX

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis
        return hw[sbox[plaintext_byte ^ key_byte]]


class HammingWeightSboxOHLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HAMMING_WEIGHT_SBOX_OH

    def __init__(self, conf):
        super().__init__(conf)
        self.onehot_outputs = 9
        self.num_outputs = (self.num_outputs[0], self.onehot_outputs)

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis
        return int_to_one_hot(hw[sbox[plaintext_byte ^ key_byte]], self.onehot_outputs)


class HammingWeightMaskedSboxLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HAMMING_WEIGHT_MASKED_SBOX

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis
        mask_byte = trace.mask[subkey_start_index]
        return hw[sbox[plaintext_byte ^ key_byte] ^ mask_byte]


class SboxLeakageModel(LeakageModel):  # No Hamming weight assumption
    leakage_type = LeakageModelType.SBOX

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis
        return sbox[plaintext_byte ^ key_byte]


class SboxOHLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.SBOX_OH

    def __init__(self, conf):
        super().__init__(conf)
        self.onehot_outputs = 256
        self.num_outputs = (self.num_outputs[0], self.onehot_outputs)

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis
        return int_to_one_hot(sbox[plaintext_byte ^ key_byte], num_classes=self.onehot_outputs)


class KeyLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.KEY

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis
        return key_byte


class KeyOHLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.KEY_OH

    def __init__(self, conf):
        super().__init__(conf)
        self.onehot_outputs = 256
        self.num_outputs = (self.num_outputs[0], self.onehot_outputs)

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis
        return int_to_one_hot(key_byte, num_classes=self.onehot_outputs)


class KeyHWLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.KEY_HW

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis
        return hw[key_byte]


class KeyHWOHLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.KEY_HW_OH

    def __init__(self, conf):
        super().__init__(conf)
        self.onehot_outputs = 9
        self.num_outputs = (self.num_outputs[0], self.onehot_outputs)

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis
        return int_to_one_hot(hw[key_byte], num_classes=self.onehot_outputs)


class KeyBitsLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.KEY_BITS

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 8)

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis

        return [(key_byte & 0x01) >> 0,
                (key_byte & 0x02) >> 1,
                (key_byte & 0x04) >> 2,
                (key_byte & 0x08) >> 3,
                (key_byte & 0x10) >> 4,
                (key_byte & 0x20) >> 5,
                (key_byte & 0x40) >> 6,
                (key_byte & 0x80) >> 7,
                ]


class KeyBitLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.KEY_BIT

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis

        return key_byte & 0x01


class AESTestLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.AES_MULTI_TEST

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 3)

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis

        return [hw[sbox[plaintext_byte ^ key_byte]],
                hw[sbox[plaintext_byte ^ key_byte]],
                hw[sbox[plaintext_byte ^ key_byte]],
                ]


class AESMultiLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.AES_MULTI

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 11)

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis

        return [hw[key_byte & 0x01],
                hw[key_byte & 0x02],
                hw[key_byte & 0x04],
                hw[key_byte & 0x08],
                hw[key_byte & 0x10],
                hw[key_byte & 0x20],
                hw[key_byte & 0x40],
                hw[key_byte & 0x80],
                hw[key_byte & 0xf0],
                hw[key_byte & 0x0f],
                hw[key_byte],
                ]


class AESBitsExLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.AES_BITS_EX

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 9)

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis

        return [(key_byte & 0x01) >> 0,
                (key_byte & 0x02) >> 1,
                (key_byte & 0x04) >> 2,
                (key_byte & 0x08) >> 3,
                (key_byte & 0x10) >> 4,
                (key_byte & 0x20) >> 5,
                (key_byte & 0x40) >> 6,
                (key_byte & 0x80) >> 7,
                0.5,
                ]


class HMACBitsLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HMAC_BITS

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 16)

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis
        key_byte_36 = key_byte ^ 0x36

        return [(key_byte & 0x01) >> 0,
                (key_byte & 0x02) >> 1,
                (key_byte & 0x04) >> 2,
                (key_byte & 0x08) >> 3,
                (key_byte & 0x10) >> 4,
                (key_byte & 0x20) >> 5,
                (key_byte & 0x40) >> 6,
                (key_byte & 0x80) >> 7,
                (key_byte_36 & 0x01) >> 0,
                (key_byte_36 & 0x02) >> 1,
                (key_byte_36 & 0x04) >> 2,
                (key_byte_36 & 0x08) >> 3,
                (key_byte_36 & 0x10) >> 4,
                (key_byte_36 & 0x20) >> 5,
                (key_byte_36 & 0x40) >> 6,
                (key_byte_36 & 0x80) >> 7,
                ]


class HMACHWLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HMAC_HW

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 3)
        self.subkey_size = 4

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        # plaintext_byte = trace.plaintext[subkey_start_index]
        key_word_list = trace.key[subkey_start_index*self.subkey_size:(subkey_start_index+1)*self.subkey_size] if key_hypothesis is None else key_hypothesis
        key_word = struct.unpack("<I", bytearray(key_word_list))[0]

        # return hw32(key_word)
        return [
            hw32(key_word),
            hw32(key_word ^ 0x36363636),
            hw32(key_word ^ 0x5c5c5c5c),
        ]


class HMACLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HMAC

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 4)

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[subkey_start_index]
        key_byte = trace.key[subkey_start_index] if key_hypothesis is None else key_hypothesis
        key_byte_36 = key_byte ^ 0x36
        key_byte_5c = key_byte ^ 0x5c

        return [key_byte,
                key_byte_36,
                key_byte,
                key_byte_5c,
                ]


class HMACHWOHLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HMAC_HW_OH

    def __init__(self, conf):
        super().__init__(conf)
        self.onehot_outputs = 33
        self.num_outputs = (self.num_outputs[0], self.onehot_outputs)
        self.subkey_size = 4

    def get_trace_leakages(self, trace, subkey_start_index, key_hypothesis=None):
        key_word_list = trace.key[subkey_start_index * self.subkey_size:(subkey_start_index + 1) * self.subkey_size] if key_hypothesis is None else key_hypothesis
        key_word = struct.unpack("<I", bytearray(key_word_list))[0]
        return int_to_one_hot(hw32(key_word), num_classes=self.onehot_outputs)
