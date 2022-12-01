from abc import ABC, abstractmethod


class BaseUncertaintyCalculator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def calculate(self, file, debug=False):
        pass
