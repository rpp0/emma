#!/usr/bin/python

"""
This program compresses datasets captured with emcap using either PCA or an autoencoder. It looks for a manifest.emcap
inside the dataset directory and applies the compression to the trace set given as argument. The program is called from
emcap to mitigate the fact that Python 3 is not supported by GNU Radio (required for calling EMMA dsp and ML functions.
After release of GNU Radio 3.8, the compress_dataset function can be directly applied on the numpy array in emcap itself.
"""

import argparse
import os
import pickle
import emma.io.io as emio
from emma.processing import ops
from emma.utils.utils import EMMAException, conf_delete_action
from emma.io.emresult import EMResult
from emma.processing.action import Action


def compress_trace_set(trace_set_path):
    if trace_set_path.endswith('.npy'):
        parent_dataset_path = os.path.dirname(trace_set_path)
        manifest_path = os.path.join(parent_dataset_path, 'manifest.emcap')

        if os.path.exists(manifest_path):
            # Open manifest
            with open(manifest_path, 'rb') as manifest_file:
                manifest = pickle.load(manifest_file)
                conf = manifest['conf']

            # Load trace set
            trace_set = emio.get_trace_set(trace_set_path, 'cw', remote=False)
            conf_delete_action(conf, 'optimize_capture')  # Make sure there is no optimize_capture action anymore

            # Add appropriate actions
            if 'pca' in manifest:
                conf.actions.append(Action('pca[%s]' % manifest_path))
            elif 'autoenc' in manifest:
                conf.actions.append(Action('corrtest[autoenc]'))

            # Perform compression
            result = EMResult()
            ops.process_trace_set(result, trace_set, conf, keep_trace_sets=True)
            processed_trace_set = result.trace_sets[0]

            # Save compressed trace set
            processed_trace_set.save(os.path.abspath(parent_dataset_path), dry=False)
        else:
            raise EMMAException("No manifest.emcap in %s, so don't know how to compress." % parent_dataset_path)
    else:
        raise EMMAException("Not a valid traceset_path in numpy format")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EMCap compress')
    parser.add_argument('trace_set_path', type=str, help="Trace set to compress")
    args = parser.parse_args()

    compress_trace_set(args.trace_set_path)
