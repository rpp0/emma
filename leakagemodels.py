# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import numpy as np
from emutils import EMMAException, int_to_one_hot
from lut import hw, sbox


class LeakageModelType:
    """
    Class that defines all possible types of leakages. Leakage model classes must have an attribute 'leakage_type' with
    one of the values defined in this class.
    """
    NONE = 'none'
    NONE_OH = 'none_oh'
    HAMMING_WEIGHT_SBOX = 'hamming_weight_sbox'
    HAMMING_WEIGHT_SBOX_OH = 'hamming_weight_sbox_oh'
    HAMMING_WEIGHT_MASKED_SBOX = 'hamming_weight_masked_sbox'
    SBOX = 'sbox'
    SBOX_OH = 'sbox_oh'
    AES_MULTI_TEST = 'aes_test'
    AES_MULTI = 'aes_multi'
    AES_BITS = 'aes_bits'
    AES_BITS_EX = 'aes_bits_ex'
    HMAC_BITS = 'hmac_bits'
    HMAC_HAMMING_WEIGHT = 'hmac_hamming_weight'
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
        self.num_outputs = (conf.key_high - conf.key_low, 1)  # [num_key_bytes, num_outputs_per_key_byte]

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

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        raise NotImplementedError

    def get_trace_set_leakages(self, trace_set):
        """
        Return numpy array containing the leakage model for [trace_index, key_byte_index]
        :param trace_set:
        :return:
        """
        values = np.zeros((len(trace_set.traces), *self.num_outputs), dtype=float)  # [num_traces, num_key_bytes]

        for i in range(len(trace_set.traces)):
            for j in range(self.num_outputs[0]):
                values[i, j] = self.get_trace_leakages(trace_set.traces[i], j + self.conf.key_low)

        return values.reshape((len(trace_set.traces), -1))


class HammingWeightSboxLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HAMMING_WEIGHT_SBOX

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        return hw[sbox[plaintext_byte ^ key_byte]]


class HammingWeightSboxOHLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HAMMING_WEIGHT_SBOX_OH

    def __init__(self, conf):
        super().__init__(conf)
        self.onehot_outputs = 9
        self.num_outputs = (self.num_outputs[0], self.onehot_outputs)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        return int_to_one_hot(hw[sbox[plaintext_byte ^ key_byte]], self.onehot_outputs)


class HammingWeightMaskedSboxLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HAMMING_WEIGHT_MASKED_SBOX

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        mask_byte = trace.mask[key_byte_index]
        return hw[sbox[plaintext_byte ^ key_byte] ^ mask_byte]


class SboxLeakageModel(LeakageModel):  # No Hamming weight assumption
    leakage_type = LeakageModelType.SBOX

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        return sbox[plaintext_byte ^ key_byte]


class SboxOHLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.SBOX_OH

    def __init__(self, conf):
        super().__init__(conf)
        self.onehot_outputs = 256
        self.num_outputs = (self.num_outputs[0], self.onehot_outputs)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        return int_to_one_hot(sbox[plaintext_byte ^ key_byte], num_classes=self.onehot_outputs)


class NoLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.NONE

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        return key_byte / 255.0


class NoOHLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.NONE_OH

    def __init__(self, conf):
        super().__init__(conf)
        self.onehot_outputs = 256
        self.num_outputs = (self.num_outputs[0], self.onehot_outputs)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        return int_to_one_hot(key_byte, num_classes=self.onehot_outputs)


class AESTestLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.AES_MULTI_TEST

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 3)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis

        return [hw[sbox[plaintext_byte ^ key_byte]],
                hw[sbox[plaintext_byte ^ key_byte]],
                hw[sbox[plaintext_byte ^ key_byte]],
                ]


class AESMultiLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.AES_MULTI

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 11)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis

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


class AESBitsLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.AES_BITS

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 8)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis

        return [(key_byte & 0x01) >> 0,
                (key_byte & 0x02) >> 1,
                (key_byte & 0x04) >> 2,
                (key_byte & 0x08) >> 3,
                (key_byte & 0x10) >> 4,
                (key_byte & 0x20) >> 5,
                (key_byte & 0x40) >> 6,
                (key_byte & 0x80) >> 7,
                ]


class AESBitsExLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.AES_BITS_EX

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 9)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis

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

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
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


class HMACHammingWeightLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HMAC_HAMMING_WEIGHT

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 4)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        key_byte_36 = key_byte ^ 0x36
        key_byte_5c = key_byte ^ 0x5c

        return [hw[key_byte],
                hw[key_byte_36],
                hw[key_byte],
                hw[key_byte_5c],
                ]


class HMACLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HMAC

    def __init__(self, conf):
        super().__init__(conf)
        self.num_outputs = (self.num_outputs[0], 4)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        key_byte_36 = key_byte ^ 0x36
        key_byte_5c = key_byte ^ 0x5c

        return [key_byte,
                key_byte_36,
                key_byte,
                key_byte_5c,
                ]
