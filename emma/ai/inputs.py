import numpy as np
from emma.utils.utils import EMMAException, int_to_one_hot, bytearray_to_many_hot
from emma.attacks.leakagemodels import LeakageModel


class AIInputType:
    """
    Class that defines all possible types of inputs for the ML models. Input classes must have an attribute
    'input_type' with one of the values defined in this class.
    """
    SIGNAL = 'signal'
    SIGNAL_PLAINTEXT = 'signal_plaintext'
    SIGNAL_PLAINTEXT_OH = 'signal_plaintext_oh'
    SIGNAL_PLAINTEXT_MH = 'signal_plaintext_mh'

    # For testing purposes
    SIGNAL_KEY = 'signal_key'
    SIGNAL_PLAINTEXT_KEY = 'signal_plaintext_key'
    PLAINTEXT_KEY = 'plaintext_key'
    PLAINTEXT_KEY_OH = 'plaintext_key_oh'
    SIGNAL_LEAKAGE = 'signal_leakage'
    RANDOM = 'random'

    @classmethod
    def choices(cls):
        """
        Get all possible AIInputTypes in list form
        :return:
        """
        c = []
        for k, v in cls.__dict__.items():
            if k[:2] != '__' and type(v) is str:
                c.append(v)
        return c


class AIInputMeta(type):
    """
    Metaclass used for checking whether the child class contains a valid input_type attribute.
    """
    class BadAIInputClassException(EMMAException):
        pass

    class InvalidInputTypeException(EMMAException):
        pass

    def __new__(mcs, name, bases, class_dict):
        if bases != (object,):  # Do not validate LeakageModel class
            if 'input_type' not in class_dict:
                raise AIInputMeta.BadAIInputClassException
            if class_dict['input_type'] not in AIInputType.choices():
                raise AIInputMeta.InvalidInputTypeException
        return type.__new__(mcs, name, bases, class_dict)


class AIInput(object, metaclass=AIInputMeta):
    """
    AI input base class.
    """
    class UnknownAIInputException(EMMAException):
        pass

    def __new__(cls, conf):
        """
        Called when instantiating an AIInput object. Returns an instance of the appropriate class depending on the
        input_type parameter.
        :param conf:
        :return:
        """
        for subclass in cls._get_subclasses():
            if subclass.input_type == conf.input_type:
                return object.__new__(subclass)  # Avoid recursion by calling object.__new__ instead of cls.__new__
        raise AIInput.UnknownAIInputException

    def __init__(self, conf):
        self.conf = conf

    @classmethod
    def _get_subclasses(cls):
        for subclass in cls.__subclasses__():
            if cls is not object:
                for subsubclass in subclass._get_subclasses():  # Also yield children of children
                    yield subsubclass
            yield subclass

    def get_trace_inputs(self, trace):
        raise NotImplementedError

    def get_trace_set_inputs(self, trace_set):
        """
        Givem a trace set, returns inputs suitable for training an AI model.
        :param trace_set:
        :return:
        """
        inputs = []

        for trace in trace_set.traces:
            inputs.append(self.get_trace_inputs(trace))

        result = np.array(inputs)

        # CNNs expect a channels dimension
        if self.conf.cnn:
            result = np.expand_dims(result, axis=-1)

        return result


class SignalAIInput(AIInput):
    input_type = AIInputType.SIGNAL

    def get_trace_inputs(self, trace):
        return trace.signal


class SignalPlaintextAIInput(AIInput):
    input_type = AIInputType.SIGNAL_PLAINTEXT

    def get_trace_inputs(self, trace):
        return np.concatenate((trace.signal, trace.plaintext))


class SignalPlaintextMHAIInput(AIInput):
    input_type = AIInputType.SIGNAL_PLAINTEXT_MH

    def get_trace_inputs(self, trace):
        return np.concatenate((trace.signal, bytearray_to_many_hot(trace.plaintext)))


class SignalPlaintextOHAIInput(AIInput):
    input_type = AIInputType.SIGNAL_PLAINTEXT_OH

    def get_trace_inputs(self, trace):
        result = []
        for p in trace.plaintext:
            result.append(int_to_one_hot(p, 256))
        result = np.concatenate(result)
        return np.concatenate((trace.signal, result))


class SignalKeyAIInput(AIInput):
    input_type = AIInputType.SIGNAL_KEY

    def get_trace_inputs(self, trace):
        return np.concatenate((trace.signal, trace.key))


class SignalPlaintextKeyAIInput(AIInput):
    input_type = AIInputType.SIGNAL_PLAINTEXT_KEY

    def get_trace_inputs(self, trace):
        return np.concatenate((trace.signal, trace.plaintext, trace.key))


class PlaintextKeyAIInput(AIInput):
    input_type = AIInputType.PLAINTEXT_KEY

    def get_trace_inputs(self, trace):
        return np.concatenate((trace.plaintext, trace.key))


class PlaintextKeyOHAIInput(AIInput):
    input_type = AIInputType.PLAINTEXT_KEY_OH

    def get_trace_inputs(self, trace):
        result = []
        for p in trace.plaintext:
            result.append(int_to_one_hot(p, 256))
        for k in trace.key:
            result.append(int_to_one_hot(k, 256))
        return np.concatenate(result)


class SignalLeakageAIInput(AIInput):
    input_type = AIInputType.SIGNAL_LEAKAGE

    def __init__(self, conf):
        super().__init__(conf)
        self.leakage_model = LeakageModel(conf)

    def get_trace_inputs(self, trace):
        leakages = []

        for k in range(16):
            leakage = self.leakage_model.get_trace_leakages(trace, k)
            if isinstance(leakage, list) or isinstance(leakage, np.ndarray):
                leakages.extend(list(leakage))
            else:
                leakages.append(leakage)
        leakages = np.array(leakages)

        return np.concatenate((trace.signal, leakages))


class RandomInput(AIInput):
    input_type = AIInputType.RANDOM

    def get_trace_inputs(self, trace):
        return np.random.uniform(0.0, 1.0, len(trace.signal))
