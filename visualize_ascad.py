#!/usr/bin/python

from ascad.ASCAD_train_models import load_ascad
import matplotlib.pyplot as plt
import argparse
import numpy as np
import code


def plot_signal(signal):
    x = np.arange(len(signal))
    plt.plot(x, signal)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ASCAD database visualizer', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('database', type=str, help='Path to ASCAD.h5')
    parser.add_argument('--average', action='store_true', default=False, help='Average all traces before plotting')
    parser.add_argument('--limit', default=50, help='Number of traces to plot')
    args, unknown = parser.parse_known_args()

    train_set, attack_set, metadata_set = load_ascad(args.database, load_metadata=True)
    metadata_train, metadata_attack = metadata_set
    train_x, train_y = train_set
    attack_x, attack_y = attack_set

    if args.average:
        mean_signal = np.mean(train_x, axis=0)
        plot_signal(mean_signal)
    else:
        for trace in train_x[:args.limit]:
            plot_signal(trace)

    code.interact(banner='', local=locals())
