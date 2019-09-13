#!/usr/bin/env python

"""
Experimental implementation of an EM leakage simulator for arbitrary binaries using GDMI.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import binascii
from emma.utils.utils import hamming_distance, random_bytes
from emma.io.traceset import Trace, TraceSet
from pygdbmi.gdbcontroller import GdbController, GdbTimeoutError
from collections import namedtuple, defaultdict
from os.path import join

AlgoritmSpecs = namedtuple("AlgoritmSpecs", ["executable", "method", "key_len", "plaintext_len"])
# REGS_TO_CHECK = None
REGS_TO_CHECK = ['1', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '152', '153', '154', '155', '156', '157', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '172', '176', '179', '180', '181', '183', '184', '185', '186', '187', '188', '189', '190', '192', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226']

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
    def __init__(self, binary, prog_args, method_name, registers, args):
        self.gdbmi = None
        self.binary = binary
        self.prog_args = prog_args
        self.done = None
        self.signal = None
        self.prev_register_values = None
        self.method_name = method_name
        self.args = args
        self.registers = registers

    def init(self):
        self.gdbmi = GdbController()
        self.gdbmi.write('-exec-arguments %s %s' % self.prog_args, read_response=False)
        self.gdbmi.write('-file-exec-and-symbols %s' % self.binary, read_response=False)
        self.gdbmi.write('-break-insert %s' % self.method_name, read_response=False)
        self.gdbmi.write('-exec-run', read_response=False)
        self.gdbmi.write('-data-list-register-names', read_response=False)

    def run(self):
        self.init()

        self.prev_register_values = {}
        self.signal = []
        self.done = False
        step_count = 0
        check_interval = 100
        register_value_interval = self.args.register_check_interval

        while not self.done:
            # print("\rStep: %d                     " % step_count, end='')

            # Parse reponses from issues commands
            if step_count % check_interval == 0:
                self.parse_responses(register_values_cb=self.update_power_consumption)

            # Send command to get register values
            if step_count % register_value_interval == 0:
                self.get_register_values(self.registers)

            # Send command to get next step
            self.program_step()
            step_count += 1

        self.gdbmi.exit()
        return np.array(self.signal)

    def run_find_varying_registers(self, nruns=3):
        self.register_value_sum = defaultdict(lambda: [])

        # Sum each register value during steps. Repeat nruns times.
        for n in range(0, nruns):
            print("Run %d..." % n)
            self.init()
            self.done = False
            self.register_value_history = defaultdict(lambda: [])
            while not self.done:
                self.get_register_values(self.registers)
                self.parse_responses(register_values_cb=self.compare_register_values)
                self.program_step()
            del self.gdbmi

            for key, values in self.register_value_history.items():
                self.register_value_sum[key].append(sum([int(x) for x in values]))

        # Check if there were runs with a different outcome
        normal_keys = []
        for key, values in self.register_value_sum.items():
            if len(set(values)) > 1:
                print("Found weird key %s: %s" % (key, str(values)))
            else:
                normal_keys.append(key)

        return normal_keys

    def compare_register_values(self, register_values):
        for key, value in register_values.items():
            self.register_value_history[key].append(value)

    def update_power_consumption(self, current_register_values):
        power_consumption = get_registers_power_consumption(self.prev_register_values, current_register_values)
        self.prev_register_values = current_register_values
        # print("Power consumption: %d" % power_consumption)
        self.signal.append(power_consumption)

    def parse_responses(self, register_values_cb=None):
        try:
            responses = self.gdbmi.get_gdb_response(timeout_sec=2)
        except GdbTimeoutError:
            print("ERROR: Got timeout from GDB. Exiting prematurely.")
            self.done = True
            return

        for response in responses:
            #print(response)

            # Check for register values
            payload = response['payload']
            if payload is not None:
                if 'register-values' in payload:
                    register_tuples = payload['register-values']
                    register_values = _parse_register_tuples(register_tuples)
                    register_values_cb(register_values)

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

    sim00 = ProgramSimulation(specs.executable, ("00"*specs.key_len, "00"*specs.plaintext_len), specs.method, REGS_TO_CHECK, args=args)
    sim0f = ProgramSimulation(specs.executable, ("0f"*specs.key_len, "00"*specs.plaintext_len), specs.method, REGS_TO_CHECK, args=args)
    simff = ProgramSimulation(specs.executable, ("ff"*specs.key_len, "00"*specs.plaintext_len), specs.method, REGS_TO_CHECK, args=args)
    plt.plot(sim00.run(), label="00")
    plt.plot(sim0f.run(), label="0f")
    plt.plot(simff.run(), label="ff")

    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Hamming distance to previous")
    plt.show()


def simulate_traces_random(args, train=True):
    """
    Untested. Simulate traces randomly, without artificial noise. Interesting for seeing true effect of random keys, but slow.
    :param args:
    :return:
    """
    specs = get_algorithm_specs(args.algorithm)
    key = random_bytes(specs.key_len)
    if train is False:
        print("Test set key: " + bytearray(key).hex())

    for i in range(0, args.num_trace_sets):
        traces = []
        print("\rSimulating trace set %d/%d...                          " % (i, args.num_trace_sets), end='')
        for j in range(0, args.num_traces_per_set):
            if train:
                key = random_bytes(specs.key_len)
            plaintext = random_bytes(specs.plaintext_len)
            key_string = binascii.hexlify(key).decode('utf-8')
            plaintext_string = binascii.hexlify(plaintext).decode('utf-8')

            sim = ProgramSimulation(specs.executable, (key_string, plaintext_string), specs.method, REGS_TO_CHECK, args=args)
            signal = sim.run()

            t = Trace(signal=signal, plaintext=plaintext, ciphertext=None, key=key, mask=None)
            traces.append(t)

        # Make TraceSet
        ts = TraceSet(name="sim-%s-%d" % (args.algorithm, i))
        ts.set_traces(traces)
        dataset_name = "sim-%s-%s" % (args.algorithm, args.mode)
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

        sim = ProgramSimulation(specs.executable, (key_string, plaintext_string), specs.method, REGS_TO_CHECK, args=args)
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


def simulate_find_varying_registers(args):
    specs = get_algorithm_specs(args.algorithm)

    sim = ProgramSimulation(specs.executable, ("00"*specs.key_len, "00"*specs.plaintext_len), specs.method, REGS_TO_CHECK, args=args)
    print(sim.run_find_varying_registers())


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("algorithm", type=str, choices=["aes", "hmacsha1"], help="Crypto algorithm to simulate.")
    arg_parser.add_argument("simtype", type=str, choices=["noisy", "random", "findvary"], help="Type of simulation experiment.")
    arg_parser.add_argument("mode", type=str, choices=["train", "test"], help="Test or train.")
    arg_parser.add_argument("--test", default=False, action="store_true", help="Run test simulation with plots.")
    arg_parser.add_argument("--debug", default=False, action="store_true", help="Run in debug mode.")
    arg_parser.add_argument("--granularity", type=str, choices=["instruction", "step", "next"], default="step", help="Granularity of power consumption simulation.")
    arg_parser.add_argument("--num-traces-per-set", type=int, default=256, help="Number of traces per trace set.")
    arg_parser.add_argument("--num-trace-sets", type=int, default=50, help="Number of trace sets to simulate.")
    arg_parser.add_argument("--output-directory", type=str, default="./datasets/", help="Output directory to write trace sets to.")
    arg_parser.add_argument("--mu", type=float, default=0.0, help="Gaussian noise mu parameter.")
    arg_parser.add_argument("--sigma", type=float, default=10.0, help="Gaussian noise sigma parameter.")
    arg_parser.add_argument("--suffix", type=str, default="", help="Dataset name suffix.")
    arg_parser.add_argument("--register-check-interval", type=int, default=1, help="Steps before checking registers.")
    args = arg_parser.parse_args()

    if args.test:
        test_and_plot(args)
    else:
        if args.simtype == 'noisy':
            simulate_traces_noisy(args)
        elif args.simtype == 'random':
            simulate_traces_random(args, train=True if args.mode == "train" else False)
        elif args.simtype == 'findvary':
            simulate_find_varying_registers(args)



