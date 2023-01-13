
from abc import abstractmethod
class Metrics():
    def __init__(self) -> None:
        pass

    @abstractmethod
    def compute(self):
        pass