import argparse
import configparser
import logging
import os

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
        self.config_path = config_path
        self.config_section = config_section
        self.emma_conf = {}
        super().__init__(*args, **kwargs)

        if os.path.exists(self.config_path):
            settings = configparser.ConfigParser()
            settings.read(self.config_path)
            emma_conf_tuples = settings.items(self.config_section)

            for k, v in emma_conf_tuples:
                self.emma_conf[k] = _config_string_to_type(v)
        else:
            logger.warning("%s does not exist; ignoring" % self.config_path)

    def _remove_prefix_chars(self, string):
        """
        Does conversion from '--foo-bar' to 'foo_bar'.
        :param string:
        :return:
        """
        result = string.lstrip(self.prefix_chars)
        result = result.replace('-', '_')
        return result

    def add_argument(self, *args, **kwargs):
        """
        Modified add_argument that overrides the 'default' parameter with the value in the config file if present.
        :param args:
        :param kwargs:
        :return:
        """
        for arg in args:
            arg_key = self._remove_prefix_chars(arg)
            if arg_key in self.emma_conf:
                kwargs['default'] = self.emma_conf[arg_key]  # Override default with value from config file
        super().add_argument(*args, **kwargs)

