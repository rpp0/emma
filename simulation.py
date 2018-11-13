#!/usr/bin/env python
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from emutils import hamming_distance
from pygdbmi.gdbcontroller import GdbController, GdbTimeoutError

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
    def __init__(self, binary, prog_args, method_name):
        self.gdbmi = None
        self.binary = binary
        self.prog_args = prog_args
        self.done = None
        self.signal = None
        self.prev_register_values = None
        self.method_name = method_name

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
        self.gdbmi.write('-exec-step-instruction', read_response=False, timeout_sec=0)
        # self.gdbmi.write('-exec-step', read_response=False, timeout_sec=0)
        # self.gdbmi.write('-exec-next', read_response=False, timeout_sec=0)

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


if __name__ == "__main__":
    pmk_string = "0000000000000000000000000000000000000000000000000000000000000000"
    data_string = "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
    key_string = "00000000000000000000000000000000"
    string_00 = "00000000000000000000000000000000"
    string_ff = "ffffffffffffffffffffffffffffffff"
    string_0f = "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f"

    # sim = ProgramSimulation("./experiments/hmac/sha1_prf", (pmk_string, data_string), "sha1_prf")
    sim00 = ProgramSimulation("./experiments/hmac/aes", (key_string, string_00), "aes_encrypt")
    sim0f = ProgramSimulation("./experiments/hmac/aes", (key_string, string_0f), "aes_encrypt")
    simff = ProgramSimulation("./experiments/hmac/aes", (key_string, string_ff), "aes_encrypt")
    plt.plot(sim00.run(), label="00")
    plt.plot(sim0f.run(), label="0f")
    plt.plot(simff.run(), label="ff")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("Hamming distance to previous")
    plt.show()



