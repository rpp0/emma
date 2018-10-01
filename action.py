from registry import operations


class Action:
    def __init__(self, action_string):
        self.action_string = action_string

        # Determine op and params
        params = None
        if '[' in action_string:
            op, _, params = action_string.rpartition('[')
            params = params.rstrip(']').split(',')
        else:
            op = action_string
        self.op = op
        self.params = params

        # Determine id name (used to identify folder where to store neural nets)
        translation_table = str.maketrans({
            '[': None,
            ']': None,
            ',': '-'
        })
        self.id_name = action_string.translate(translation_table)

    @classmethod
    def get_actions_from_conf(cls, conf):
        actions = []
        for action_string in conf.actions:
            actions.append(Action(action_string))
        return actions
