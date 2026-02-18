"""
This will be my very own warning module. I'm using the same type of classes as Exceptions, with the difference that these don't disturb runtime.
"""

from ..utils.textformatting import warning_emoji

# Parent class
class Warning:
    def __init__(self, statement: str):
        print(f'{warning_emoji} {self.__class__.__name__}: {statement}')