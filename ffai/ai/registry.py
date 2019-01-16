"""
==========================
Author: Niels Justesen
Year: 2019
==========================
This module contains enumerations and tables for the rules.
"""


class BotRegistry:

    def __init__(self):
        self.bots = {}

    def register(self, id, cls):
        if id in self.bots:
            raise Exception('Bot with ID {} already registered.'.format(id))
        self.bots[id] = cls

    def make(self, id):
        if id not in self.bots:
            raise Exception('Bot with ID {} not registered.'.format(id))
        return self.bots[id](id)


# Have a global registry
registry = BotRegistry()


def register_bot(id, cls):
    return registry.register(id, cls)


def make_bot(id):
    return registry.make(id)
