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
            if 'leakage_type' not in class_dict or 'num_outputs' not in class_dict:
                raise LeakageModelMeta.BadLeakageModelClassException
            if class_dict['leakage_type'] not in LeakageModelType.choices():
                raise LeakageModelMeta.InvalidModelTypeException
            if type(class_dict['num_outputs']) is not tuple:
                raise LeakageModelMeta.InvalidModelTypeException
            if len(class_dict['num_outputs']) < 1:
                raise LeakageModelMeta.InvalidModelTypeException
        return type.__new__(mcs, name, bases, class_dict)


class LeakageModel(object, metaclass=LeakageModelMeta):
    """
    Leakage model base class.
    """
    num_outputs = None

    class UnknownLeakageModelException(EMMAException):
        pass

    def __new__(cls, leakage_type):
        """
        Called when instantiating a LeakageModel object. Returns an instance of the appropriate class depending on the
        leakage_type parameter.
        :param leakage_type:
        :return:
        """
        for subclass in cls._get_subclasses():
            if subclass.leakage_type == leakage_type:
                return object.__new__(subclass)  # Avoid recursion by calling object.__new__ instead of cls.__new__
        raise LeakageModel.UnknownLeakageModelException

    @classmethod
    def _get_subclasses(cls):
        for subclass in cls.__subclasses__():
            if cls is not object:
                for subsubclass in subclass._get_subclasses():  # Also yield children of children
                    yield subsubclass
            yield subclass

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        raise NotImplementedError

    def get_trace_set_leakages(self, trace_set):
        values = np.zeros((len(trace_set.traces), *self.__class__.num_outputs), dtype=float)  # [num_traces, num_key_bytes]

        for i in range(len(trace_set.traces)):
            for j in range(self.__class__.num_outputs[0]):
                values[i, j] = self.get_trace_leakages(trace_set.traces[i], j)

        return values


class HammingWeightSboxLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HAMMING_WEIGHT_SBOX
    num_outputs = (16,)  # TODO Number of key bytes should not be specified here. Rather, in conf passed to LeakageModel

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        return hw[sbox[plaintext_byte ^ key_byte]]


class HammingWeightMaskedSboxLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.HAMMING_WEIGHT_MASKED_SBOX
    num_outputs = (16,)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        mask_byte = trace.mask[key_byte_index]
        return hw[sbox[plaintext_byte ^ key_byte] ^ mask_byte]


class SboxLeakageModel(LeakageModel):  # No Hamming weight assumption
    leakage_type = LeakageModelType.HAMMING_WEIGHT_SBOX
    num_outputs = (16,)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        plaintext_byte = trace.plaintext[key_byte_index]
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        return sbox[plaintext_byte ^ key_byte]


class NoLeakageModel(LeakageModel):
    leakage_type = LeakageModelType.NONE
    num_outputs = (16,)

    def get_trace_leakages(self, trace, key_byte_index, key_hypothesis=None):
        key_byte = trace.key[key_byte_index] if key_hypothesis is None else key_hypothesis
        return key_byte / 255.0
