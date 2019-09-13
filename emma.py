#!/usr/bin/env python3

"""
Main script for performing EM side-channel analysis. Sends processing tasks (ops) to the broker in order to complete
a given activity.
"""

from emma_worker import app
from celery.utils.log import get_task_logger
from emma.utils.configargumentparser import ConfigArgumentParser
from emma.attacks.leakagemodels import LeakageModelType
from emma.ai.inputs import AIInputType
from emma.utils.utils import conf_has_op, EMMAConfException, get_default_keras_loss_names
from emma.processing.action import Action

import argparse
import subprocess
import emma.io.io
from emma.utils import utils, registry

logger = get_task_logger(__name__)


def args_epilog():
    """
    Build epilog for the help instructions of EMMA.
    """
    result = "Actions can take the following parameters between square brackets ('[]'):\n"
    for op in registry.operations.keys():
        result += "{:>20s} ".format(op)
        if op in registry.operations_optargs:
            result += "["
            for optarg in registry.operations_optargs[op]:
                result += "{:s}, ".format(optarg)
            result = result.strip().rstrip(',')
            result += "]"
        result += "\n"
    return result


def clear_redis():
    """
    Clear any previous results from Redis. Sadly, there is no cleaner way atm.
    """
    try:
        subprocess.check_output(["redis-cli", "flushall"])
        logger.info("Redis cleared")
    except (FileNotFoundError, subprocess.CalledProcessError):
        logger.warning("Could not clear local Redis database")


class EMMAHost:
    def __init__(self, args):
        args.actions = Action.get_actions_from_conf(args)  # TODO dirty fix. The problem is datasets are being passed as string
        self.dataset, self.dataset_ref, self.dataset_val = self._get_datasets(args)
        self.conf = self._generate_conf(args)

    @staticmethod
    def _get_datasets(args):
        # Load dataset from worker node
        dataset = emma.io.io.get_dataset(dataset=args.dataset, conf=args, remote=args.remote)

        # Load reference set if applicable. Otherwise just use a reference from the standard dataset
        if args.refset is not None:
            dataset_ref = emma.io.io.get_dataset(dataset=args.refset, conf=args, remote=args.remote)
        else:
            dataset_ref = dataset

        # Load validation set if applicable
        if args.valset is not None:
            dataset_val = emma.io.io.get_dataset(dataset=args.valset, conf=args, remote=args.remote)
        else:
            dataset_val = None

        return dataset, dataset_ref, dataset_val

    def _resolve_conflicts(self, conf):
        # Overrides
        if conf_has_op(conf, 'rwindow'):
            conf.max_cache = 0  # rwindow randomly shifts traces, so we cannot cache these traces during training

        # Sanity checks
        if conf.refset and not conf_has_op(conf, 'align'):
            raise EMMAConfException("Refset specified, but no align action")
        if conf.key_low >= conf.key_high:
            raise EMMAConfException("key_low should be < key_high")
        if self.dataset_val is not None and str(self.dataset.id) == str(self.dataset_val.id):
            raise EMMAConfException("Validation set should never be the same as the training set.")
        if conf_has_op(conf, 'keyplot'):
            if not conf_has_op(conf, 'groupkeys'):  # Transparently add a groupkeys op, which is required for keyplot
                conf.actions = conf.actions[0:-1] + [Action('groupkeys')] + conf.actions[-1:]
        if conf_has_op(conf, 'corrtrain'):
            if conf.loss_type == 'softmax_crossentropy' and conf.key_high - conf.key_low != 1:
                raise EMMAConfException("Softmax crossentropy with multiple key bytes is currently not supported")
        if conf_has_op(conf, 'classify') and ('oh' not in conf.leakage_model):
            raise EMMAConfException("Leakage model for classify op should contain 'oh'.")

    def _generate_conf(self, args):
        if self.dataset is None or self.dataset_ref is None:
            raise Exception("Tried to generate configuration without loading datasets.")

        conf = argparse.Namespace(
            format=self.dataset.format,
            reference_signal=self.dataset_ref.reference_signal,
            traces_per_set=self.dataset.traces_per_set,
            datasets_path=self.dataset.root,
            dataset_id=self.dataset.id,
            subkey=0,
            **args.__dict__
        )

        self._resolve_conflicts(conf)

        return conf

    def _determine_activity(self):  # TODO put under activities.py
        num_activities = 0
        activity = None
        params = None

        for action in self.conf.actions:
            if action.op in registry.activities.keys():
                activity = registry.activities[action.op]
                params = action.params
                num_activities += 1

        if num_activities > 1:
            raise Exception("Only one activity can be executed at a time. Choose from: %s" % str(
                registry.activities.keys()))

        if activity is None:
            activity = registry.activities['default']
        if params is None:
            params = []

        return activity, params

    def run(self):
        activity, params = self._determine_activity()
        activity(self, *params)


if __name__ == "__main__":
    parser = ConfigArgumentParser(description='Electromagnetic Mining Array (EMMA)', epilog=args_epilog(), formatter_class=argparse.RawDescriptionHelpFormatter, config_section='EMMA')
    parser.add_argument('actions', type=str, help='Action to perform. Choose from %s' % str(registry.operations.keys()), nargs='+')
    parser.add_argument('dataset', type=str, help='Identifier of dataset to use')
    parser.add_argument('--outform', dest='outform', type=str, choices=['cw', 'sigmf', 'gnuradio'], default='cw', help='Output format to use when saving')
    parser.add_argument('--max-subtasks', type=int, default=32, help='Maximum number of subtasks')
    parser.add_argument('--key-low', type=int, default=2, help='Low index of subkeys to break.')
    parser.add_argument('--key-high', type=int, default=3, help='High index of subkeys to break.')
    parser.add_argument('--kill-workers', default=False, action='store_true', help='Kill workers after finishing the tasks.')
    parser.add_argument('--butter-type', type=str, default='lowpass', help='Default type of Butterworth filter')
    parser.add_argument('--butter-order', type=int, default=1, help='Default order of Butterworth filter')
    parser.add_argument('--butter-cutoff', type=float, default=0.01, help='Default cutoff of Butterworth filter')
    parser.add_argument('--butter-fs', type=int, default=None, help='Default sampling frequency of Butterworth filter')
    parser.add_argument('--windowing-method', type=str, default='rectangular', help='Windowing method')
    parser.add_argument('--hamming', default=False, action='store_true', help='Use Hamming weight instead of true byte values.')
    parser.add_argument('--augment-roll', default=False, action='store_true', help='Roll signal during data augmentation.')
    parser.add_argument('--augment-noise', default=False, action='store_true', help='Add noise to the signal during data augmentation.')
    parser.add_argument('--augment-shuffle', default=True, action='store_true', help='Shuffle examples randomly along first axis.')  # Do not disable
    parser.add_argument('--update', default=False, action='store_true', help='Update existing AI model instead of replacing.')
    parser.add_argument('--online', default=False, action='store_true', help='Fetch samples from remote EMcap instance online (without storing to disk).')
    parser.add_argument('--remote', default=False, action='store_true', help='Send processing tasks to remote Celery workers for faster processing.')
    parser.add_argument('--local', dest='remote', action='store_false', help='')
    parser.add_argument('--refset', type=str, default=None, help='Dataset to take reference signal from for alignment (default = same as dataset argument)')
    parser.add_argument('--valset', type=str, default=None, help='Dataset to take validation set traces from (default = same as dataset argument)')
    parser.add_argument('--model-suffix', type=str, default=None, help='Suffix for model name.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--max-cache', type=int, default=None, help='Max trace sets in cache. Default: all.')
    parser.add_argument('--num-valsets', type=int, default=64, help='Training split of cross-validation trace sets to use (if none specified with --valset)')
    parser.add_argument('--max-num-tracesets', type=int, default=None, help='Maximum number of trace set paths to load from all datasets (train, ref and val).')
    parser.add_argument('--normalize', default=False, action='store_true', help='Normalize input data before feeding to NN')
    parser.add_argument('--tfold', default=False, action='store_true', help='Train using t-fold cross-validation')
    parser.add_argument('--n-hidden-layers', type=int, default=1, help='Number of hidden layers in model.')
    parser.add_argument('--n-hidden-nodes', type=int, default=256, help='Number of hidden nodes per hidden layer in model.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
    parser.add_argument('--activation', type=str, default='leakyrelu', help='Activation function of model.')
    parser.add_argument('--cnn', default=False, action='store_true', help='Use ASCAD CNN for AICorrNet')
    parser.add_argument('--norank', default=False, action='store_true', help='Do not calculate rank')
    parser.add_argument('--testrank', default=False, action='store_true', help='Load model and test rank for varying test set sizes.')
    parser.add_argument('--regularizer', type=str, default=None, help='Regularizer to use.')
    parser.add_argument('--reglambda', type=float, default=0.001, help='Regularizer lambda.')
    parser.add_argument('--leakage-model', type=str, choices=LeakageModelType.choices(), default=LeakageModelType.HAMMING_WEIGHT_SBOX, help='Assumed leakage model.')
    parser.add_argument('--input-type', type=str, choices=AIInputType.choices(), default=AIInputType.SIGNAL, help='Type of input to use for training ML models.')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size.')
    parser.add_argument('--metric-freq', type=int, default=10, help='Frequency of calculating metrics (e.g. rank) of model.')
    parser.add_argument('--use-bias', default=True, action='store_true', help='Use a bias term.')  # TODO: It's impossible to disable this now; fix
    parser.add_argument('--batch-norm', default=True, action='store_true', help='Use batch normalization.')  # TODO: It's impossible to disable this now; fix
    parser.add_argument('--saliency-remove-bias', default=False, action='store_true', help='Remove first samples when using the salvis activity.')
    parser.add_argument('--saliency-mean-gradient', default=True, action='store_true', help='Get the mean gradient of the batch instead of individual gradients when visualizing saliency.')  # TODO: Impossible to disable
    parser.add_argument('--saliency-num-traces', type=int, default=1024, help='Maxmimum number of traces to show in saliency plots.')
    parser.add_argument('--loss-type', type=str, choices=list(
        registry.lossfunctions.keys()) + get_default_keras_loss_names(), default='correlation', help='Loss function to use when training.')
    parser.add_argument('--plot-no-reference', default=False, action='store_true', help='Do not plot reference signal.')
    parser.add_argument('--plot-num-traces', type=int, default=1024, help='Maxmimum number of traces to show in plots.')
    parser.add_argument('--plot-title', type=str, default='', help='Title of the plot.')
    parser.add_argument('--plot-xlabel', type=str, default='', help='Xlabel of the plot.')
    parser.add_argument('--plot-ylabel', type=str, default='', help='Ylabel of the plot.')
    parser.add_argument('--plot-colorbar-label', type=str, default='', help='Colorbar label of the plot.')
    parser.add_argument('--specgram-samprate', type=int, default=8000000, help='Sample rate (used in axis calculation for specgram)')
    parser.add_argument('--plot-force-timedomain', default=False, action='store_true', help='Force time domain x label')
    args, unknown = parser.parse_known_args()
    print(utils.BANNER)

    if len(unknown) > 0:
        raise EMMAConfException("Unknown arguments: %s" % str(unknown))

    try:
        clear_redis()

        emma = EMMAHost(args)
        emma.run()
    except KeyboardInterrupt:
        pass

    # Clean up
    print("Cleaning up")
    app.control.purge()
    app.backend.cleanup()
    if args.kill_workers:
        subprocess.check_output(['pkill', '-9', '-f', 'celery'])
