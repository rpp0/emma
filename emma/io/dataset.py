import configparser
import emma.io.io
from os import listdir
from os.path import isfile, join
from emma.utils.utils import conf_has_op, EMMAConfException


class Dataset:
    def __init__(self, id, dataset_conf, emma_conf=None):
        self.id = id
        self.dataset_conf = dataset_conf
        self.format = dataset_conf["format"]
        self.reference_index = int(dataset_conf["reference_index"])
        self.traces_per_set = 0

        self._setup(emma_conf)

    def _setup(self, emma_conf):
        """
        Get a list of relative trace set paths for the dataset identifier and retrieve
        a reference signal for the entire dataset.

        Example trace set paths:
        em-arduino/trace1.npy
        em-arduino/trace2.npy
        ...
        em-arduino/tracen.npy

        Where trace1.npy is loaded as the reference signal.

        At a later time, the relative paths need to be resolved to absolute paths
        on the workers.
        """
        settings = configparser.RawConfigParser()
        settings.read('settings.conf')
        self.root = settings.get("Datasets", "datasets_path")

        # Assign trace set paths
        if self.format == "cw":  # .npy
            path = join(self.root, self.id)
            self.trace_set_paths = sorted([join(self.id, f) for f in listdir(path) if isfile(join(path, f)) and '_traces.npy' in f])
        elif self.format == "sigmf":  # .meta
            self.trace_set_paths = None
            raise NotImplementedError
        elif self.format == "gnuradio":  # .cfile
            self.trace_set_paths = None
            raise NotImplementedError
        elif self.format == "ascad":  # ASCAD .h5
            if ':' not in self.id:
                raise EMMAConfException("No group specified. Specify the H5 group to use by using a colon, e.g. file:group")
            file, _, group = self.id.rpartition(":")
            path = join(self.root, 'ASCAD/ASCAD_data/ASCAD_databases/%s.h5' % file)

            # Make sure we never use training set when attacking or classifying
            self.trace_set_paths = emma.io.io.get_ascad_paths(path, group)
        else:
            raise Exception("Unknown input format '%s'" % self.format)

        assert(len(self.trace_set_paths) > 0)

        # Assign reference signal
        reference_trace_set = emma.io.io.get_trace_set(join(self.root, self.trace_set_paths[0]), self.format, ignore_malformed=False, remote=False)  # TODO add parameter to allow choosing reference trace set index. Fixed now to 0.

        self.traces_per_set = len(reference_trace_set.traces)
        self.reference_signal = reference_trace_set.traces[self.reference_index].signal


def get_dataset_normalization_mean_std(name):
    """
    Statistics precomputed with get_dataset_statistics.py
    :param name:
    :return:
    """
    if name == 'em-corr-arduino' or name == 'em-cpa-arduino':
        mean = 0.014595353784991782
        std = 0.006548281541447703
    elif name == 'ASCAD':
        mean = -11.587280595238095
        std = 25.75363459386104
    elif name == 'ASCAD_desync50':
        mean = -11.195121833333333
        std = 25.89963055607876
    elif name == 'ASCAD_desync100':
        mean = -11.093145738095238
        std = 26.11483790582092
    else:
        return 0.0, 1.0

    return mean, std
