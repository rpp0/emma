#!/usr/bin/python3
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Worker node using Celery
# Copyright 2017, Pieter Robyns
# ----------------------------------------------------

from __future__ import absolute_import
from celery import Celery
import configparser

settings = configparser.RawConfigParser()  # TODO: error handling, detect first usage of EMMA by presence of settings.conf
settings.read('settings.conf')
broker = settings.get("Network", "broker")
backend = settings.get("Network", "backend")

app = Celery('emma',
             broker=broker,
             backend=backend,
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
