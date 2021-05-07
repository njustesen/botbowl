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
        if id.lower() in self.bots:
            raise Exception('Bot with ID {} already registered.'.format(id.lower()))
        self.bots[id.lower()] = cls

    def make(self, id):
        if id.lower() not in self.bots:
            raise Exception('Bot with ID {} not registered.'.format(id.lower()))
        return self.bots[id.lower()](id.lower())

    def list(self):
        result = []
        for key in self.bots:
            result.append(key)
        return result


# Have a global registry
registry = BotRegistry()


def register_bot(id, cls):
    return registry.register(id, cls)


def make_bot(id):
    return registry.make(id)


def list_bots():
    return registry.list()
