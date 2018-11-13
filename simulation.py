#!/usr/bin/env python
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import argparse
import numpy as np
import matplotlib.pyplot as plt
import binascii
from emutils import hamming_distance, random_bytes
from traceset import Trace, TraceSet
from pygdbmi.gdbcontroller import GdbController, GdbTimeoutError
from collections import namedtuple
from os.path import join

AlgoritmSpecs = namedtuple("AlgoritmSpecs", ["executable", "method", "key_len", "plaintext_len"])

# TODO: There are some commented prints in here; these should be refactored to log in DEBUG


def _parse_register_value(register_value):
    try:
        return int(register_value, 16)
    except ValueError:
        # print("Warning: failed to parse register value '%s'. Substituting by 0." % register_value)
        return 0


def _parse_register_tuples(register_tuples):
    register_values = {}

    for register_tuple in register_tuples:
        register_id = register_tuple['number']
        register_value = register_tuple['value']
        register_values[register_id] = _parse_register_value(register_value)

    return register_values


def get_registers_power_consumption(previous_registers, current_registers):
    total_power_consumption = 0

    for register_id in current_registers:
        if register_id in previous_registers.keys():
            hd = hamming_distance(previous_registers[register_id], current_registers[register_id])
            # print("%d -> %d: %d" % (previous_registers[register_id], current_registers[register_id], hd))
        else:
            hd = hamming_distance(0, current_registers[register_id])
            # print("%d -> %d: %d" % (0, current_registers[register_id], hd))
        total_power_consumption += hd

    return total_power_consumption


class ProgramSimulation:
    def __init__(self, binary, prog_args, method_name, args):
        self.gdbmi = None
        self.binary = binary
        self.prog_args = prog_args
        self.done = None
        self.signal = None
        self.prev_register_values = None
        self.method_name = method_name
        self.args = args

    def run(self):
        self.gdbmi = GdbController()
        self.gdbmi.write('-exec-arguments %s %s' % self.prog_args, read_response=False)
        self.gdbmi.write('-file-exec-and-symbols %s' % self.binary, read_response=False)
        self.gdbmi.write('-break-insert %s' % self.method_name, read_response=False)
        self.gdbmi.write('-exec-run', read_response=False)
        self.gdbmi.write('-data-list-register-names', read_response=False)

        self.prev_register_values = {}
        self.signal = []
        self.done = False
        step_count = 0
        check_interval = 100
        register_value_interval = 1
        filter_list = None
        # filter_list = [str(x) for x in range(0, 40)]
        while not self.done:
            print("\rStep: %d                     " % step_count, end='')

            # Parse reponses from issues commands
            if step_count % check_interval == 0:
                self.parse_responses()

            # Send command to get register values
            if step_count % register_value_interval == 0:
                self.get_register_values(filter_list)

            # Send command to get next step
            self.program_step()
            step_count += 1

        self.gdbmi.exit()
        return np.array(self.signal)

    def update_power_consumption(self, current_register_values):
        power_consumption = get_registers_power_consumption(self.prev_register_values, current_register_values)
        self.prev_register_values = current_register_values
        # print("Power consumption: %d" % power_consumption)
        self.signal.append(power_consumption)

    def parse_responses(self):
        try:
            responses = self.gdbmi.get_gdb_response(timeout_sec=2)
        except GdbTimeoutError:
            print("ERROR: Got timeout from GDB. Exiting prematurely.")
            self.done = True
            return

        for response in responses:
            print(response)

            # Check for register values
            payload = response['payload']
            if payload is not None:
                if 'register-values' in payload:
                    register_tuples = payload['register-values']
                    register_values = _parse_register_tuples(register_tuples)
                    self.update_power_consumption(register_values)

            # Check for end packet
            if 'type' in response and response['type'] == 'notify':
                if response['message'] == 'thread-exited':
                    self.done = True

    def program_step(self):
        """
        Step program
        :return:
        """
        if self.args.granularity == 'instruction':
            self.gdbmi.write('-exec-step-instruction', read_response=False, timeout_sec=0)
        elif self.args.granularity == 'step':
            self.gdbmi.write('-exec-step', read_response=False, timeout_sec=0)
        elif self.args.granularity == 'next':
            self.gdbmi.write('-exec-next', read_response=False, timeout_sec=0)

    def get_register_values(self, target_registers=None):
        # Filter?
        if target_registers is not None:
            register_list = ' '.join(target_registers)
        else:
            register_list = ''

        self.gdbmi.write('-data-list-register-values r %s' % register_list, read_response=False, timeout_sec=0)

    def get_changed_registers(self):
        """
        DEPRECATED
        Get list of changed registers. Not used anymore because just batching requests for all register values
        is faster than checking which ones changed, waiting, and then querying for them.
        :return:
        """
        self.gdbmi.write('-data-list-changed-registers', read_response=False)


def get_algorithm_specs(algorithm):
    if algorithm == "aes":
        return AlgoritmSpecs(executable="./experiments/simulate/aes", method="aes_encrypt", key_len=16, plaintext_len=16)
    elif algorithm == "hmacsha1":
        return AlgoritmSpecs(executable="./experiments/simulate/sha1_prf", method="sha1_prf", key_len=32, plaintext_len=76)


def test_and_plot(args):
    specs = get_algorithm_specs(args.algorithm)

    sim00 = ProgramSimulation(specs.executable, ("00"*specs.key_len, "00"*specs.plaintext_len), specs.method, args=args)
    sim0f = ProgramSimulation(specs.executable, ("0f"*specs.key_len, "00"*specs.plaintext_len), specs.method, args=args)
    simff = ProgramSimulation(specs.executable, ("ff"*specs.key_len, "00"*specs.plaintext_len), specs.method, args=args)
    plt.plot(sim00.run(), label="00")
    plt.plot(sim0f.run(), label="0f")
    plt.plot(simff.run(), label="ff")

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Hamming distance to previous")
    plt.show()


def simulate_traces_random(args):
    """
    Untested. Simulate traces randomly, without artificial noise. Interesting for seeing true effect of random keys, but slow.
    :param args:
    :return:
    """
    specs = get_algorithm_specs(args.algorithm)

    for i in range(0, args.num_trace_sets):
        traces = []
        print("\rSimulating trace set %d/%d...                          " % (i, args.num_trace_sets), end='')
        for j in range(0, args.num_traces_per_set):
            key = random_bytes(specs.key_len)
            plaintext = random_bytes(specs.plaintext_len)
            key_string = binascii.hexlify(key).decode('utf-8')
            plaintext_string = binascii.hexlify(plaintext).decode('utf-8')

            sim = ProgramSimulation(specs.executable, (key_string, plaintext_string), specs.method, args=args)
            signal = sim.run()

            t = Trace(signal=signal, plaintext=plaintext, ciphertext=None, key=key, mask=None)
            traces.append(t)

        # Make TraceSet
        ts = TraceSet(name="sim-%s-%d" % (args.algorithm, i))
        ts.set_traces(traces)
        dataset_name = "sim-%s" % args.algorithm
        ts.save(join(args.output_directory, dataset_name + args.suffix))


def simulate_traces_noisy(args):
    specs = get_algorithm_specs(args.algorithm)

    key = random_bytes(specs.key_len)
    for i in range(0, 256):
        print("\rSimulating noisy trace sets for key %d...      " % i, end='')
        key[2] = i
        plaintext = random_bytes(specs.plaintext_len)
        key_string = binascii.hexlify(key).decode('utf-8')
        plaintext_string = binascii.hexlify(plaintext).decode('utf-8')

        sim = ProgramSimulation(specs.executable, (key_string, plaintext_string), specs.method, args=args)
        signal = sim.run()

        traces = []
        for j in range(0, args.num_traces_per_set):
            mod_signal = signal + np.random.normal(args.mu, args.sigma, len(signal))
            t = Trace(signal=mod_signal, plaintext=plaintext, ciphertext=None, key=key, mask=None)
            traces.append(t)

            # Debug
            if args.debug:
                plt.plot(mod_signal)
                plt.show()

        # Make TraceSet
        ts = TraceSet(name="sim-noisy-%s-%d" % (args.algorithm, i))
        ts.set_traces(traces)
        dataset_name = "sim-noisy-%s" % args.algorithm
        ts.save(join(args.output_directory, dataset_name + args.suffix))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("algorithm", type=str, choices=["aes", "hmacsha1"], help="Crypto algorithm to simulate.")
    arg_parser.add_argument("simtype", type=str, choices=["noisy", "random"], help="Type of simulation experiment.")
    arg_parser.add_argument("--test", default=False, action="store_true", help="Run test simulation with plots.")
    arg_parser.add_argument("--debug", default=False, action="store_true", help="Run in debug mode.")
    arg_parser.add_argument("--granularity", type=str, choices=["instruction", "step", "next"], default="step", help="Granularity of power consumption simulation.")
    arg_parser.add_argument("--num-traces-per-set", type=int, default=256, help="Number of traces per trace set.")
    arg_parser.add_argument("--num-trace-sets", type=int, default=200, help="Number of trace sets to simulate.")
    arg_parser.add_argument("--output-directory", type=str, default="./datasets/", help="Output directory to write trace sets to.")
    arg_parser.add_argument("--mu", type=float, default=0.0, help="Gaussian noise mu parameter.")
    arg_parser.add_argument("--sigma", type=float, default=10.0, help="Gaussian noise sigma parameter.")
    arg_parser.add_argument("--suffix", type=str, default="", help="Dataset name suffix.")
    args = arg_parser.parse_args()

    if args.test:
        test_and_plot(args)
    else:
        if args.simtype == 'noisy':
            simulate_traces_noisy(args)
        elif args.simtype == 'random':
            simulate_traces_random(args)




