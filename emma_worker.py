#!/usr/bin/env python3

"""
Spawns Celery worker nodes for performing distributed operations on a set of traces.
"""

from __future__ import absolute_import
from celery import Celery
import configparser
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Ignore Tensorflow deprecation and performance warnings
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Do not use GPU

try:
    settings = configparser.RawConfigParser()
    settings.read('settings.conf')
    broker = settings.get("Network", "broker")
    backend = settings.get("Network", "backend")
except FileNotFoundError:
    print("No settings.conf found! Please create it before running EMMA.")
    exit(1)

app = Celery('emma',
             broker=broker,
             backend=backend,
             include=['emma.processing.ops', 'emma.processing.action', 'ascad'])

# Optional configuration, see the application user guide.
app.conf.update(
    task_serializer='pickle',
    task_compression='zlib',
    accept_content={'pickle'},
    result_serializer='pickle',
    # worker_max_tasks_per_child=1
)

if __name__ == '__main__':
    app.start()
