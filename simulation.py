#!/usr/bin/env python
# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from emutils import EMMAException, hamming_distance
from pygdbmi.gdbcontroller import GdbController

# TODO: There are some commented prints in here; these should be refactored to log in DEBUG


def simulate_trace_set(simulation_type):
    if simulation_type == "sim-hmac":
        return _simulate_hmac_trace_set()
    else:
        raise EMMAException("Unknown simulation type '%s'" % simulation_type)


def _simulate_hmac_trace_set():
    pass


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
    def __init__(self, binary, prog_args):
        self.gdbmi = GdbController()
        self.binary = binary
        self.prog_args = prog_args

    def run(self):
        print(self.gdbmi.write('-exec-arguments %s %s' % self.prog_args))
        print(self.gdbmi.write('-file-exec-and-symbols %s' % self.binary))
        print(self.gdbmi.write('-break-insert sha1_prf'))
        print(self.gdbmi.write('-exec-run'))
        print(self.gdbmi.write('-data-list-register-names'))

        prev_register_values = {}
        signal = []
        done = False
        step_count = 0
        check_interval = 100
        while not done:
            print("\rStep: %d                     " % step_count, end='')

            if step_count % check_interval == 0:
                # Get changed registers and their values
                changed_registers = self.get_changed_registers()
                register_values = self.get_register_values(changed_registers)

                # We have register values that changed
                if register_values is not None:
                    power_consumption = get_registers_power_consumption(prev_register_values, register_values)
                    prev_register_values = register_values
                    # print("Power consumption: %d" % power_consumption)
                    signal.append(power_consumption)

            # Go to next instruction
            done = self.program_step()
            step_count += 1

        return np.array(signal)

    def program_step(self):
        """
        Step program and return True of the program has finished running.
        :return:
        """
        responses = self.gdbmi.write('-exec-step-instruction')
        # responses = self.gdbmi.write('-exec-next')

        for response in responses:
            if 'type' in response and response['type'] == 'notify':
                if response['message'] == 'thread-exited':
                    return True

        return False

    def get_register_values(self, target_registers):
        if target_registers:  # Make sure we have any registers to check
            register_list = ' '.join(target_registers)
            responses = self.gdbmi.write('-data-list-register-values r %s' % register_list)
            for response in responses:
                payload = response['payload']
                register_tuples = payload['register-values']
                return _parse_register_tuples(register_tuples)

    def get_changed_registers(self):
        responses = self.gdbmi.write('-data-list-changed-registers')

        for response in responses:
            try:
                payload = response['payload']
                return payload['changed-registers']
            except TypeError:
                # print("Payload is none")
                pass
            except KeyError:
                # print("No changed-registers")
                pass

        return None


if __name__ == "__main__":
    pmk_string = "0000000000000000000000000000000000000000000000000000000000000000"
    data_string = "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"

    args = (pmk_string, data_string)
    sim = ProgramSimulation("./experiments/hmac/sha1_prf", args)
    nruns = 1
    for i in range(0, nruns):
        result = sim.run()
        plt.plot(result)
    plt.show()



