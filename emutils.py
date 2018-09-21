# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017-2018, Pieter Robyns
# ----------------------------------------------------

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


def get_action_op_params(action_string):
    """
    Convert action string to (op, parameters) tuple.
    :param action_string:
    :return:
    """
    params = None
    if '[' in action_string:
        op, _, params = action_string.rpartition('[')
        params = params.rstrip(']').split(',')
    else:
        op = action_string

    return op, params


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
    models that use a certain configuration to separate directories.
    :param conf:
    :return:
    """
    conf_dict = conf.__dict__
    result = ""
    first = True
    translation_table = str.maketrans({
        '[': None,
        ']': None,
        ',': '-'
    })

    if 'actions' in conf_dict:
        for action in conf_dict['actions'][:-1]:
            if first:
                first = False
            else:
                result += "-"
            result += action.translate(translation_table)
    if 'dataset_id' in conf_dict:
        result += "-" + conf_dict['dataset_id']

    # Ignore corrtest in naming because it's something between action and activity TODO make this cleaner somehow
    result = result.replace("-corrtest", "")

    return result


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
