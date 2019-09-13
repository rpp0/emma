#!/usr/bin/env python

"""
Script to convert dataset from "Screaming Channels: When Electromagnetic Side Channels Meet Radio Transceivers" by
Camurati et al. to the ChipWhisperer file format.
"""

import sys
import argparse
import configparser
import logging
import os
import binascii
import numpy as np
from datetime import datetime

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger(__name__)


def set_if_none(target, value):
    if target is None:
        return value
    else:
        raise Exception("Tried to override value of %s, which is supposed to be None." % target)


def get_sc_files(path):
    trace_files = []
    key_file = None
    plaintexts_file = None

    file_listing = os.listdir(path)
    for f in file_listing:
        f_path = os.path.join(path, f)
        if os.path.isfile(f_path):
            if ".npy" in f:
                trace_files.append(f_path)
            elif ".txt" in f:
                if f.startswith("key"):
                    key_file = set_if_none(key_file, f_path)
                elif f.startswith("pt"):
                    plaintexts_file = set_if_none(plaintexts_file, f_path)

    trace_sorter = lambda x: int(x.rpartition("_")[2].partition(".")[0])
    return sorted(trace_files, key=trace_sorter), key_file, plaintexts_file


def str_to_np(string):
    return np.array(bytearray(binascii.unhexlify(string.strip())), dtype=np.uint8)


def save_batch(traces, key, plaintexts, count, dest_path, dry=False):
    end = count+1
    begin = end - len(traces)

    np_traces = np.array(traces)
    np_plaintexts = np.array(plaintexts[begin:end])
    np_keys = np.array([key]*len(np_plaintexts))

    if len(np_keys) != len(np_traces) or len(np_plaintexts) != len(np_traces):
        raise Exception("Non-matching sizes in batch")

    # print(np_traces)
    # print(np_keys)
    # print(np_plaintexts)
    date_str = str(datetime.utcnow()).replace(" ", "_").replace(".", "_").replace(":", "-")
    filename = "sc_avg_%d_%d_%s" % (begin, end, date_str)
    logger.info("Saving batch [%d:%d] to %s" % (begin, end, os.path.join(dest_path, "%s_*.npy" % filename)))
    if not dry:
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        np.save(os.path.join(dest_path, "%s_traces.npy" % filename), np_traces)
        np.save(os.path.join(dest_path, "%s_knownkey.npy" % filename), np_keys)
        np.save(os.path.join(dest_path, "%s_textin.npy" % filename), np_plaintexts)

    traces.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert "Screaming Channels" datasets to ChipWhisperer format.')
    parser.add_argument("sc_path", type=str, help="Path to the Screaming Channels dataset.")
    parser.add_argument("--traces_per_traceset", type=int, default=256, help="Number of traces to put in a single trace set.")
    parser.add_argument("--dry", default=False, action="store_true", help="Do not effectively store traces.")
    args = parser.parse_args()

    # Open settings file and read location of EMMA datasets
    settings = configparser.RawConfigParser()
    with open("settings.conf", "r") as f:
        settings.read_file(f)
    datasets_path = settings.get("Datasets", "datasets_path")
    sc_name = args.sc_path.rpartition("/")[2].rstrip("/")

    # Start conversion
    logger.info("Converting dataset %s from %s to %s" % (sc_name, args.sc_path.partition(sc_name)[0], datasets_path))

    trace_files, key_file, plaintexts_file = get_sc_files(args.sc_path)
    traces = []
    keys = []
    plaintexts = []
    key = None

    # Read key
    with open(key_file, "r") as f:
        key = str_to_np(f.read())

    # Read plaintexts
    logger.info("Reading %d plaintexts" % len(trace_files))
    with open(plaintexts_file, "r") as f:
        plaintext_lines = f.readlines()[0:len(trace_files)]
        for line in plaintext_lines:
            plaintexts.append(str_to_np(line))
        plaintexts = np.array(plaintexts)

    # Read batches of traces
    for i, trace_file in enumerate(trace_files):
        if str(i) not in trace_file:
            raise Exception("Count number is not in trace file!")
        trace = np.load(trace_file)
        traces.append(trace)
        if len(traces) == args.traces_per_traceset:
            save_batch(traces, key, plaintexts, i, os.path.join(datasets_path, sc_name), dry=args.dry)
    save_batch(traces, key, plaintexts, i, os.path.join(datasets_path, sc_name), dry=args.dry)


