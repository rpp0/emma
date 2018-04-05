#!/usr/bin/python

from emio import get_trace_set
from dsp import butter_filter
import matplotlib.pyplot as plt

trace_set = get_trace_set("/run/media/pieter/ext-drive/em-cpa-arduino/2018-03-29_15:41:47_658357_traces.npy", format="cw")

for trace in trace_set.traces:
    plt.plot(butter_filter(trace.signal))
plt.show()
