#!/usr/bin/python3
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Worker node using Celery
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

from __future__ import absolute_import
from celery import Celery

app = Celery('emma',
             #broker='redis://:password@mini:6379/0',
             broker='redis://localhost:6379/0',
             backend='redis://localhost:6379/0',
             include=['ops'])

# Optional configuration, see the application user guide.
app.conf.update(
    result_expires=60,
    task_serializer='pickle',
    accept_content={'pickle'},
    result_serializer='pickle'
)

if __name__ == '__main__':
    app.start()
