from abc import ABC, abstractmethod


class IDemo(ABC):
    def __init__(self):
        self.running = False
        self.name = "Abstract Demo Class"
        pass

    @abstractmethod
    def start(self):
        pass

    def announceStart(self):
        print(f"Now running {self.name}...\n")
