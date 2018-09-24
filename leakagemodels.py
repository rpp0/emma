# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import numpy as np
from emutils import EMMAException
from lut import hw, sbox


class LeakageModelType:
    """
    Class that defines all possible types of leakages. Leakage model classes must have an attribute 'leakage_type' with
    one of the values defined in this class.
    """
    NONE = 'none'
    HAMMING_WEIGHT_SBOX = 'hamming_weight_sbox'
    HAMMING_WEIGHT_MASKED_SBOX = 'hamming_weight_masked_sbox'
    SBOX = 'sbox'
    AES_TEST = 'aes_test'

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


class HammingWeightMaskedSboxLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HAMMING_WEIGHT_MASKED_SBOX

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        mask_byte = trace.mask[key_byte_index]
        return hw[sbox[plaintext_byte ^ key_byte] ^ mask_byte]


class SboxLeakageModel(LeakageModel):  # No Hamming weight assumption
    leakage_type = LeakageModelType.HAMMING_WEIGHT_SBOX

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        return sbox[plaintext_byte ^ key_byte]


class NoLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.NONE

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        return key_byte / 255.0


class AESMultiLeakageTestModel(LeakageModel):
    leakage_type = LeakageModelType.AES_TEST

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
