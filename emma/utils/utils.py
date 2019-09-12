import numpy as np
import socket
import fcntl
import struct

BANNER = """  _____ __  __ __  __    _
 | ____|  \/  |  \/  |  / \\
 |  _| | |\/| | |\/| | / _ \\
 | |___| |  | | |  | |/ ___ \\
 |_____|_|  |_|_|  |_/_/   \_\\
 |Electromagnetic Mining Array
 ============================="""


def chunks(input_list, chunk_size):
    """
    Divide a list into chunks of size 'chunk_size'.
    :param input_list:
    :param chunk_size:
    :return:
    """
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i+chunk_size]


def partition(input_list, num_partitions):
    """
    Divide list in 'num_partitions' partitions.
    :param input_list:
    :param num_partitions:
    :return:
    """
    n = int(len(input_list) / num_partitions)

    for i in range(0, len(input_list), n):
        yield input_list[i:i+n]


def numpy_to_hex(np_array):
    """
    Convert numpy array to hex offset.
    :param np_array:
    :return:
    """
    result = ""
    for elem in np_array:
        result += "{:0>2} ".format(hex(elem)[2:])
    return result


def pretty_print_subkey_scores(np_array, limit_rows=20, descending=True):
    """
    Print score matrix as a nice table.
    :param np_array:
    :param limit_rows:
    :param descending:
    :return:
    """
    if type(np_array) != np.ndarray:
        raise TypeError("Expected np.ndarray")
    elif len(np_array.shape) != 2:
        raise ValueError("Expected 2D array")
    else:
        print('')
        num_subkeys = np_array.shape[0]
        num_guess_values = np_array.shape[1]

        # Sort array
        sorted_scores = []
        for subkey in range(0, num_subkeys):
            sorted_subkey = sorted(zip(np_array[subkey, :], range(num_guess_values)), key=lambda f: f[0], reverse=descending)[0:limit_rows]
            sorted_scores.append(sorted_subkey)

        # Print header
        for subkey in range(0, num_subkeys):
            print("    {:>2d}      ".format(subkey), end='')
        print("\n" + "-"*192)

        # Print body
        for key_guess in range(0, limit_rows):
            for subkey in range(0, num_subkeys):
                score, byte = sorted_scores[subkey][key_guess]
                print(" {:>4.2f} ({:02x}) |".format(float(score), byte), end='')
            print('')


# Source: https://stackoverflow.com/questions/24196932/how-can-i-get-the-ip-address-of-eth0-in-python
def get_ip_address(ifname):
    """
    Gets the IP address of an interface.
    :param ifname:
    :return:
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', bytes(ifname[:15], encoding='utf-8'))
    )[20:24])


def conf_to_id(conf):
    """
    Converts an EMMA configuration object to a string that represents an ID for the experiment. Useful for storing
    models that use a certain configuration to separate directories. The directory name is based on the concatenation
    of the id_name parameters of the provided actions in the command line arguments.
    :param conf:
    :return:
    """
    conf_dict = conf.__dict__
    result = ""
    first = True

    if 'actions' in conf_dict:
        for action in conf_dict['actions'][:-1]:
            if action.id_name is None or action.id_name == "":  # Skip empty action
                continue

            if first:
                first = False
            else:
                result += "-"
            result += action.id_name
    # if 'dataset_id' in conf_dict:
    #     result += "-" + conf_dict['dataset_id']

    return result


def conf_has_op(conf, target_op):
    """
    Checks whether target_action is in conf.actions, ignoring params
    :param conf:
    :param target_op:
    :return:
    """
    for action in conf.actions:
        if target_op == action.op:
            return True
    return False


def conf_get_action(conf, target_op):
    """
    TODO: Make dedicated conf object instead of dictionary
    Returns actions(s) with op name target_op from conf.
    :param conf:
    :param target_op:
    :return:
    """
    result = []
    for action in conf.actions:
        if target_op == action.op:
            result.append(action)

    if len(result) == 0:
        return None
    elif len(result) == 1:
        return result[0]
    else:
        return result


def conf_delete_action(conf, target_op):
    """
    TODO: Make dedicated conf object instead of dictionary
    Returns actions(s) with op name target_op from conf.
    :param conf:
    :param target_op:
    :return:
    """
    for action in conf.actions:
        if target_op == action.op:
            target = action
            break

    conf.actions.remove(target)


def shuffle_random_multiple(lists):
    """
    Shuffle n same-length lists in the same random order.
    :param lists:
    :return:
    """
    result = []
    if len(lists) < 1:
        raise EMMAException("shuffle_random_xy expects at least 1 array")

    length = len(lists[0])
    random_indices = np.arange(length)
    np.random.shuffle(random_indices)

    for l in lists:
        if len(l) != length:
            raise EMMAException("Array length %d != %d. Expecting same-size lists." % (len(l), length))
        shuffled_l = np.take(l, random_indices, axis=0)
        result.append(shuffled_l)

    del lists
    return result


def int_to_one_hot(integer, num_classes):
    """
    Convert integer to one-hot vector with num_classes elements.
    :param integer:
    :param num_classes:
    :return:
    """
    r = np.zeros(num_classes, dtype=np.float32)
    r[integer] = 1.0
    return r


def bytearray_to_many_hot(array):
    """
    Convert byte array to many-hot vector
    :param array:
    :return:
    """
    masks = [2**7, 2**6, 2**5, 2**4, 2**3, 2**2, 2**1, 1]
    bit_vector = []
    for a in array:
        for mask in masks:
            bit_vector.append((a & mask) > 0)
    return np.array(bit_vector, dtype=np.uint8)


def get_default_keras_loss_names():
    """
    Get the default available losses from Keras and return them as list of strings.
    :return:
    """
    return ['categorical_crossentropy']


def hamming_distance(v1, v2):
    """
    Get Hamming distance between integers v1 and v2.
    :param v1:
    :param v2:
    :return:
    """
    return bin(v1 ^ v2).count("1")


def random_bytes(n):
    """
    Get n random bytes from /dev/urandom and return as numpy array of uint8.
    :param n:
    :return:
    """

    result = None
    with open("/dev/urandom", "rb") as f:
        result = f.read(n)

    return np.array(bytearray(result))


def binary_to_hex(binary: bytearray):
    """
    Convert an array of bytes to offset hex string notation.
    :param binary:
    :return:
    """
    result = []
    for elem in binary:
        result.append("%02x" % elem)
    return ' '.join(result)


class Window(object):
    """
    Helper object for specifying a range between begin (inclusive) and end (exclusive).
    """
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        if end is not None and begin is not None:
            self.size = end - begin
        else:
            self.size = None


class EMMAException(Exception):
    pass


class EMMAConfException(EMMAException):
    pass


class MaxPlotsReached(EMMAException):
    pass
