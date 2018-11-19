# -*- coding: utf-8 -*-
class DictList(dict):
    def __missing__(self, key):
        self.__setitem__(key, [])
        return self[key]


# LAZY PROPERTY EVALUATION
class Lazy(object):
    def __init__(self, calculate_function):
        self._calculate = calculate_function

    def __get__(self, obj, _=None):
        if obj is None:
            return self
        value = self._calculate(obj)
        setattr(obj, self._calculate.func_name, value)
        return value
