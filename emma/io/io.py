import numpy as np
import emma.processing.ops as ops
import configparser
import pickle
import h5py
from emma.io.traceset import TraceSet
from emma.io.dataset import Dataset
from emma.utils.utils import Window, EMMAConfException
from os.path import join


def get_dataset(dataset, conf=None, remote=True):
    """
    Wrapper function for getting dataset properties, either from the local or remote worker.
    :param dataset:
    :param conf:
    :param remote: Override for configured remote parameter.
    :return:
    """
    if remote:
        return ops.remote_get_dataset.si(dataset, conf=conf).apply_async().get()
    else:
        return _get_dataset(dataset, conf)


def get_trace_set(trace_set_path, format, ignore_malformed=True, remote=True):
    """
    Wrapper function for getting a trace set, either from the local or remote worker.
    :param trace_set_path:
    :param format:
    :param ignore_malformed:
    :param remote: Override for configured remote parameter.
    :return:
    """
    if remote:
        return ops.remote_get_trace_set.si(trace_set_path, format, ignore_malformed).apply_async().get()
    else:
        return _get_trace_set(trace_set_path, format, ignore_malformed)


def _get_dataset(dataset_id, conf=None):
    """
    Retrieve the dataset properties (trace sets, reference index to use, etc.) from the local
    node for a given dataset_id.
    """
    datasets_conf = configparser.RawConfigParser()
    datasets_conf.read('datasets.conf')

    # Does identifier exist?
    dataset_name = dataset_id.rpartition(":")[0]
    if dataset_name in datasets_conf.sections():
        return Dataset(dataset_id, dataset_conf=datasets_conf[dataset_name], emma_conf=conf)
    else:
        raise Exception("Dataset %s does not exist in datasets.conf" % dataset_id)


def _get_trace_set(trace_set_path, format, ignore_malformed=True):
    """
    Load traces in from absolute path trace_set_path into a TraceSet object depending on the format.
    """

    if format == "cw":
        name = trace_set_path.rpartition('_traces')[0]
        plaintext_set_path = name + '_textin.npy'
        ciphertext_set_path = name + '_textout.npy'
        key_set_path = name + '_knownkey.npy'

        existing_properties = []
        try:
            traces = np.load(trace_set_path, encoding="bytes", allow_pickle=True)
            existing_properties.append(traces)
        except FileNotFoundError:
            traces = None

        try:
            plaintexts = np.load(plaintext_set_path, encoding="bytes", allow_pickle=True)
            existing_properties.append(plaintexts)
        except FileNotFoundError:
            print("WARNING: No plaintext for trace %s" % name)
            plaintexts = None

        try:
            ciphertexts = np.load(ciphertext_set_path, encoding="bytes", allow_pickle=True)
            existing_properties.append(ciphertexts)
        except FileNotFoundError:
            ciphertexts = None

        try:
            keys = np.load(key_set_path, encoding="bytes", allow_pickle=True)
            existing_properties.append(keys)
        except FileNotFoundError:
            keys = np.array([[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]]*traces.shape[0])
            print("No key file found! Using 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15")
            #keys = None

        masks = None  # No masks for Arduino experiments

        if ignore_malformed:  # Discard malformed traces
            for property in existing_properties:
                if traces.shape[0] != property.shape[0]:
                    return None

            return TraceSet(name=name, traces=traces, plaintexts=plaintexts, ciphertexts=ciphertexts, keys=keys, masks=masks)
        else:  # Just truncate malformed traces instead of discarding
            if not traces is None:
                traces = traces[0:len(plaintexts)]
            if not ciphertexts is None:
                ciphertexts = ciphertexts[0:len(plaintexts)]
            if not keys is None:
                keys = keys[0:len(plaintexts)]
            if not masks is None:
                masks = masks[0:len(plaintexts)]

            return TraceSet(name=name, traces=traces, plaintexts=plaintexts, ciphertexts=ciphertexts, keys=keys, masks=masks)
    elif format == "sigmf":  # .meta
        raise NotImplementedError
    elif format == "gnuradio":  # .cfile
        raise NotImplementedError
    elif format == "ascad":
        return get_ascad_trace_set(trace_set_path)
    else:
        print("Unknown trace input format '%s'" % format)
        exit(1)

    return None


def get_ascad_paths(path, group, split=256):
    with h5py.File(path, "r") as h5file:
        h5group = h5file[group]
        h5datasets = h5group.keys()

        # Some sanity checks
        if 'labels' not in h5datasets or 'metadata' not in h5datasets or 'traces' not in h5datasets:
            raise EMMAConfException("H5 dataset %s does not contain labels, metadata and traces" % path)

        if len(h5group['labels']) != len(h5group['traces']) or len(h5group['metadata']) != len(h5group['traces']):
            raise EMMAException("Labels, traces and metadata in %s are not the same size" % path)

        dataset_size = len(h5group['traces'])

        # Make list of segmented trace paths
        return ["%s#%s[%d:%d]" % (path, group, i, min(i+split, dataset_size)) for i in range(0, dataset_size, split)]


def get_ascad_trace_set(uri):
    """
    Given a URI, convert ASCAD data to a TraceSet object.
    """
    trace_set = None

    # Process URI
    path, _, group_subset = uri.rpartition("#")
    group, _, index = group_subset.rpartition("[")
    index = index.rstrip("]")
    min_index, _, max_index = index.rpartition(":")
    min_index = int(min_index)
    max_index = int(max_index)

    with h5py.File(path, "r") as h5file:
        h5group = h5file[group]

        traces = h5group["traces"][min_index:max_index]
        plaintexts = h5group["metadata"][min_index:max_index]["plaintext"]
        keys = h5group["metadata"][min_index:max_index]["key"]
        masks = h5group["metadata"][min_index:max_index]["masks"]

        traces = np.array(traces)
        plaintexts = np.array(plaintexts)
        keys = np.array(keys)
        masks = np.array(masks)

        trace_set = TraceSet(name=uri, traces=traces, plaintexts=plaintexts, ciphertexts=None, keys=keys, masks=masks)
        trace_set.window = Window(begin=0, end=len(trace_set.traces[0].signal))
        trace_set.windowed = True

    return trace_set


def update_cw_config(path, trace_set, update_dict):
    """
    Update ChipWhisperer config file in order to reflect changes made to
    the traces by EMMA.
    """
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


def write_emcap_manifest(conf, pca):
    # TODO: PCA should be saved in the same way as an AI model, not this way
    data = {
        "conf": conf,
        "pca": pca,
    }

    print("Writing manifest...")
    manifest_dst = '/tmp/manifest.emcap'
    with open(manifest_dst, 'wb') as f:
        pickle.dump(data, f)
    print("Done. Manifest saved at %s" % manifest_dst)
