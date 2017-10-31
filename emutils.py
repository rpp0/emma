# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

import numpy as np

BANNER = """  _____ __  __ __  __    _
 | ____|  \/  |  \/  |  / \\
 |  _| | |\/| | |\/| | / _ \\
 | |___| |  | | |  | |/ ___ \\
 |_____|_|  |_|_|  |_/_/   \_\\
 |Electromagnetic Mining Array
 ============================="""

def partition(input_list, partition_size):
    '''
    Partition a list into chunks of size 'partition_size'
    '''
    for i in range(0, len(input_list), partition_size):
        yield input_list[i:i + partition_size]

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
        sorted_correlations = []
        for subkey in range(0, 16):
            sorted_subkey = sorted(zip(np_array[subkey,:], range(256)), key=lambda f: f[0], reverse=True)[0:limit_rows]
            sorted_correlations.append(sorted_subkey)

        for subkey in range(0, 16):
            print("    {:>2d}      ".format(subkey), end='')
        print("\n" + "-"*192)
        for key_guess in range(0, limit_rows):
            for subkey in range(0, 16):
                corr, byte = sorted_correlations[subkey][key_guess]
                print(" {:>4.2f} ({:02x}) |".format(float(corr), byte),end='')
            print('')

class Window(object):
    def __init__(self, begin, end):
        self.begin = begin
        self.end = end
        self.size = end - begin
