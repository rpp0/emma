# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

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
