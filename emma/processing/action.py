from emma.utils.registry import operations_id_overrides


class Action:
    """
    action: 'window[0,900]'
    op: 'window'
    optargs: '[0,900]'
    action.id_name: 'window0-900'
    """

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
        if op in operations_id_overrides:
            self.id_name = operations_id_overrides[op]
        else:
            translation_table = str.maketrans({
                '[': None,
                ']': None,
                ',': '-'
            })
            self.id_name = action_string.translate(translation_table)

    def __repr__(self):
        return "Action('%s')" % self.action_string

    @classmethod
    def get_actions_from_conf(cls, conf):
        actions = []
        for action_string in conf.actions:
            actions.append(Action(action_string))
        return actions
