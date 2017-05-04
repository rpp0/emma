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

def pretty_print_table(np_array, limit_rows=20):
    if type(np_array) != np.ndarray:
        print("Warning: pretty_print_table: not a numpy array!")
        return
    elif len(np_array.shape) != 2:
        print("Warning: pretty_print_table: not a 2D numpy array!")
        return
    else:
        # Sort array
        sorted_array = []
        for i in range(0, np_array.shape[1]):
            sorted_array.append(sorted(np_array[:,i], reverse=True))
        sorted_array = np.transpose(np.matrix(sorted_array))

        # Print array
        tmp = np.get_printoptions()
        np.set_printoptions(threshold=np.inf, linewidth=np.inf, precision=2)
        print(sorted_array[0:20,:])
        np.set_printoptions(**tmp)
