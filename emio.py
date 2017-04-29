import numpy as np
from os import listdir
from os.path import isfile, join

def get_trace_paths(input_path, inform):
    if inform == "cw":  # .npy
        return sorted([join(input_path,f) for f in listdir(input_path) if isfile(join(input_path, f)) and '_traces.npy' in f])
    elif inform == "sigmf":  # .meta
        raise NotImplementedError
    elif inform == "gnuradio":  # .cfile
        raise NotImplementedError
    else:
        print("Unknown input format '%s'" % inform)
        exit(1)

def get_trace_set(trace_set_path, inform):  # TODO wrap in some kind of SigMF class?
    if inform == "cw":
        return np.load(trace_set_path)
    elif inform == "sigmf":  # .meta
        raise NotImplementedError
    elif inform == "gnuradio":  # .cfile
        raise NotImplementedError
    else:
        print("Unknown input format '%s'" % inform)
        exit(1)
