# ----------------------------------------------------
# Electromagnetic Mining Array (EMMA)
# Copyright 2018, Pieter Robyns
# ----------------------------------------------------

import argparse
import configparser
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _config_string_to_type(string):
    if string.lower() in ('true', 'false'):
        return bool(string)
    elif len(string) < 1:
        return None
    elif string[0] in ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'):
        try:
            if '.' in string:
                return float(string)
            else:
                return int(string)
        except ValueError:
            return None


class ConfigArgumentParser(argparse.ArgumentParser):
    """
    Wrapper class for ArgumentParser that uses an additional config file for overriding default arguments. This results
    in the following override priorities: default arguments < config arguments < CLI arguments.
    """
    def __init__(self, *args, config_path='settings.conf', config_section='DEFAULT', **kwargs):
        super().__init__(*args, **kwargs)
        self.config_path = config_path
        self.config_section = config_section

    def update_args_with_conf(self, args, given_args):
        """
        Updates args with items listed in the config file under self.config_section, unless they are in given_args.
        :param args:
        :param given_args:
        :return:
        """
        if os.path.exists(self.config_path):
            settings = configparser.ConfigParser()
            settings.read(self.config_path)
            emma_conf = settings.items(self.config_section)

            for k, v in emma_conf:
                if k in given_args:
                    continue
                setattr(args, k, _config_string_to_type(v))
        else:
            logger.warning("%s does not exist; ignoring" % self.config_path)

        return args

    def _get_given_args(self):
        """
        Hack to get optional arguments that were explicitly specified by the user.
        :return:
        """
        given_args = set()

        for action in self._actions:
            for option_string in action.option_strings:
                if option_string in sys.argv[1:]:
                    if option_string.startswith('--'):
                        given_args.add(option_string[2:].replace("-", "_"))
                    elif option_string.startswith('-'):
                        given_args.add(option_string[1:].replace("-", "_"))
                    else:
                        given_args.add(option_string.replace("-", "_"))

        return given_args

    def parse_known_args(self, *args, **kwargs):
        # Parse arguments along with defaults
        known_args, unknown_args = super().parse_known_args(*args, **kwargs)

        # Get args given by user via CLI
        given_args = self._get_given_args()

        # Update the known_args with args specified in settings.conf
        return self.update_args_with_conf(known_args, given_args), unknown_args
