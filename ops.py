import numpy as np
import sys
import matplotlib.pyplot as plt
from emma_worker import app
from dsp import *
from functools import wraps
from os.path import join, basename
from namedtuples import Window

ops = {}  # Op registry

def op(name):
    '''
    Defines the @op decorator
    '''
    def decorator(func):
        ops[name] = func
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    return decorator

@op('align')
def align_trace_set(trace_set, reference):
    aligned_trace_set = []

    for trace in trace_set:
        aligned_trace = align(trace, reference)
        if not aligned_trace is None:
            aligned_trace_set.append(aligned_trace)

    return np.array(aligned_trace_set)

@app.task
def work(trace_set_paths, args, conf):
    print("Node performing %s on trace set of length %d" % (str(args.actions), len(trace_set_paths)))

    # Perform actions on the sample sets
    for trace_set_path in trace_set_paths:
        # Load trace set depending on the format TODO
        trace_set_name = basename(trace_set_path)
        trace_set = np.load(trace_set_path)
        if conf.reference_trace is None:
            conf.reference_trace = trace_set[0][conf.window.begin:conf.window.end]

        for action in args.actions: # TODO
            # Print progress
            print("\rAligning %s...             " % trace_set_name),
            sys.stdout.flush()

            # Align trace set
            trace_set = ops['align'](trace_set, conf.reference_trace)
            #align_trace_set.delay(trace_set, conf.reference_trace)
            #trace_set = filter_trace_set(trace_set)

            # Write to output
            if args.outform == 'plot':
                for trace in trace_set:
                    plt.plot(range(0, len(trace)), butter_filter(trace))
                plt.show()
            elif args.outform == 'cw':
                # Save back to output file
                np.save(join(output_path, trace_set_name), trace_set)

                # Update the corresponding config file
                update_cw_config(output_path, trace_set, {"numPoints": len(conf.reference_trace)})
            elif args.outform == 'sigmf':
                count = 1
                for trace in trace_set:
                    trace.tofile(join(output_path_gnuradio, "%s-%d.rf32_le" % (trace_set.rpartition('_')[0], count)))
                    count += 1
            else:
                print("Unknown format: %s" % args.format)
                exit(1)
    return "Finished"
