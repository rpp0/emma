# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

import numpy as np
import configparser
import ops
import configparser
from sigmf.sigmffile import SigMFFile
from traceset import TraceSet
from dataset import Dataset

def remote_get_dataset(dataset):
    return ops.remote_get_dataset.si(dataset).apply_async().get()

def remote_get_trace_set(trace_set_path, format, ignore_malformed=True):
    return ops.remote_get_trace_set.si(trace_set_path, format, ignore_malformed).apply_async().get()

def get_dataset(dataset):
    '''
    Get a full list of relative trace set paths given a dataset ID. This is used by the EMMA
    master node to distribute the trace set paths to EMMA worker nodes.
    '''
    datasets_conf = configparser.RawConfigParser()
    datasets_conf.read('datasets.conf')

    # Does identifier exist?
    if dataset in datasets_conf.sections():
        format = datasets_conf[dataset]["format"]
        reference_index = int(datasets_conf[dataset]["reference_index"])
        return Dataset(dataset, format, reference_index)
    else:
        raise Exception("Dataset %s does not exist in datasets.conf" % dataset)

def get_trace_set(trace_set_path, format, ignore_malformed=True):
    '''
    Load traces in from absolute path trace_set_path into a TraceSet object depending on the format.
    '''

    if format == "cw":
        name = trace_set_path.rpartition('_traces')[0]
        plaintext_set_path = name + '_textin.npy'
        ciphertext_set_path = name + '_textout.npy'
        key_set_path = name + '_knownkey.npy'

        existing_properties = []
        try:
            traces = np.load(trace_set_path, encoding="bytes")  # TODO make more robust towards non-existing paths
            existing_properties.append(traces)
        except FileNotFoundError:
            traces = None

        try:
            plaintexts = np.load(plaintext_set_path, encoding="bytes")
            existing_properties.append(plaintexts)
        except FileNotFoundError:
            print("WARNING: No plaintext for trace %s" % name)
            plaintexts = None

        try:
            ciphertexts = np.load(ciphertext_set_path, encoding="bytes")
            existing_properties.append(ciphertexts)
        except FileNotFoundError:
            ciphertexts = None

        try:
            keys = np.load(key_set_path, encoding="bytes")
            existing_properties.append(keys)
        except FileNotFoundError:
            keys = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]]*traces.shape[0])
            print("No key file found! Using 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16")
            #keys = None

        if ignore_malformed:  # Discard malformed traces
            for property in existing_properties:
                if traces.shape[0] != property.shape[0]:
                    return None

            return TraceSet(name=name, traces=traces, plaintexts=plaintexts, ciphertexts=ciphertexts, keys=keys)
        else:  # Just truncate malformed traces instead of discarding
            if not traces is None:
                traces = traces[0:len(plaintexts)]
            if not ciphertexts is None:
                ciphertexts = ciphertexts[0:len(plaintexts)]
            if not keys is None:
                keys = keys[0:len(plaintexts)]

            return TraceSet(name=name, traces=traces, plaintexts=plaintexts, ciphertexts=ciphertexts, keys=keys)
    elif format == "sigmf":  # .meta
        raise NotImplementedError
    elif format == "gnuradio":  # .cfile
        raise NotImplementedError
    else:
        print("Unknown input format '%s'" % format)
        exit(1)

    return None

def update_cw_config(path, trace_set, update_dict):
    '''
    Update ChipWhisperer config file in order to reflect changes made to
    the traces by EMMA.
    '''
    cp = configparser.RawConfigParser()
    cp.optionxform = str  # Preserve case sensitivity

    # Read file
    config_file_path = join(path, 'config_' + trace_set.name + '_.cfg')
    cp.read(config_file_path)

    for key in update_dict:
        cp.get("Trace Config", key)
        cp.set("Trace Config", key, update_dict[key])

    with open(config_file_path, 'w') as config_file_path_fp:
        cp.write(config_file_path_fp)
