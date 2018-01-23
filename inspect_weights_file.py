#!/usr/bin/python

import pickle
import matplotlib.pyplot as plt
import numpy as np

a = pickle.load(open("/home/pieter/sshfs/weights.p", "rb"))
plt.plot(a)
plt.show()
