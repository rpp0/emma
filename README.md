# ElectroMagnetic Mining Array (EMMA)

EMMA is a framework for capturing and attacking traces of electromagnetic radiation emitted by an electronic device, in order to obtain encryption keys or other sensitive data.


## Installation

The recommended way of installing EMMA is via `venv`:
```bash
$ cd <emma_root_directory>
$ python -m venv env
$ source env/bin/activate
$ pip install -r requirements.txt
$ emma.py -h
```


## Configuration

Two config files should be added to the EMMA root directory: `settings.conf` and `datasets.conf`.


### `settings.conf`
#### Example configuration

```
[Network]
broker = redis://:password@redisserver:6379/0
backend = redis://:password@redisserver:6379/0

[Datasets]
datasets_path = /home/user/my-dataset-directory/
stream_interface = eth0

[EMMA]
remote = True
```


### `datasets.conf`
#### Example configuration

```
# Custom dataset (train)
[em-corr-arduino]
format=cw
reference_index=0

# Custom dataset (test)
[em-cpa-arduino]
format=cw
reference_index=0

# ASCAD database
[ASCAD]
format=ascad
reference_index=0
```


## Getting started

Although EMMA can run on a single machine, ideally at least two machines should be used to perform an analysis: a master (low-end device for sending commands) and one or more slaves (high-end devices for performing calculations). At least one device must have `redis` installed and configured. Before continuing, make sure this device is listed as the backend in `settings.conf` (see above).

On each slave that should perform computations, run the following command to automatically spawn worker processes for each available CPU:

```
$ celery -A emma_worker worker -l info -Q celery,priority.high
```

Note that each slave should be able to access the datasets listen in `datasets.conf`. Now, the master can issue commands to process these datasets.

```
$ emma.py plot ASCAD:Profiling_traces
```

Most ops can be executed locally (without distributing tasks to worker nodes):

```
$ emma.py plot ASCAD:Profiling_traces --local
```

See `emma.py -h` for a full list of available commands.


## EMcap

EMcap is a tool that allows for convenient capturing and storage of EM trace datasets using Software Defined Radios (SDRs).

See `emcap.py -h` for a full list of available commands.


## Running tools

```bash
$ cd <emma_root_directory>
$ PYTHONPATH=. tools/visualize_dataset.py em-cpa-arduino
```


## Citing

If this project was helpful to you and you wish to credit the project, you could list the following citation in your work:

Robyns, P., Quax, P. and Lamotte, W. 2018. Improving CEMA using Correlation Optimization. IACR Transactions on Cryptographic Hardware and Embedded Systems. 2019, 1 (Nov. 2018), 1-24. DOI:https://doi.org/10.13154/tches.v2019.i1.1-24.  