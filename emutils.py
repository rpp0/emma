# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
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

def chunks(input_list, chunk_size):
    '''
    Divide a list into chunks of size 'chunk_size'
    '''
    for i in range(0, len(input_list), chunk_size):
        yield input_list[i:i+chunk_size]

def partition(input_list, num_partitions):
    '''
    Divide list in 'num_partitions' partitions
    '''
    n = int(len(input_list)/ num_partitions)

    for i in range(0, len(input_list), n):
        yield input_list[i:i+n]

def numpy_to_hex(np_array):
    result = ""
    for elem in np_array:
        result += "{:0>2} ".format(hex(elem)[2:])
    return result

def pretty_print_correlations(np_array, limit_rows=20):
    if type(np_array) != np.ndarray:
        print("Warning: pretty_print_table: not a numpy array!")
        return
    elif len(np_array.shape) != 2:
        print("Warning: pretty_print_table: not a 2D numpy array!")
        return
    else:
        # Sort array
        print('')
        num_subkeys = np_array.shape[0]
        sorted_correlations = []
        for subkey in range(0, num_subkeys):
            sorted_subkey = sorted(zip(np_array[subkey,:], range(256)), key=lambda f: f[0], reverse=True)[0:limit_rows]
            sorted_correlations.append(sorted_subkey)

        for subkey in range(0, num_subkeys):
            print("    {:>2d}      ".format(subkey), end='')
        print("\n" + "-"*192)
        for key_guess in range(0, limit_rows):
            for subkey in range(0, num_subkeys):
                corr, byte = sorted_correlations[subkey][key_guess]
                print(" {:>4.2f} ({:02x}) |".format(float(corr), byte),end='')
            print('')

class Window(object):
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        if not end is None and not begin is None:
            self.size = end - begin
        else:
            self.size = None

# Source: https://stackoverflow.com/questions/24196932/how-can-i-get-the-ip-address-of-eth0-in-python
def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,  # SIOCGIFADDR
        struct.pack('256s', bytes(ifname[:15], encoding='utf-8'))
    )[20:24])

def conf_to_id(conf):
    conf_dict = conf.__dict__
    result = ""
    first = True
    translation_table = str.maketrans({
        '[': None,
        ']': None,
        ',': '-'
    })

    if 'actions' in conf_dict:
        for action in conf_dict['actions']:
            if first:
                first = False
            else:
                result += "-"
            result += action.translate(translation_table)
    if 'dataset_id' in conf_dict:
        result += "-" + conf_dict['dataset_id']

    return result
