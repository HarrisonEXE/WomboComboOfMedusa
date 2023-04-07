from abc import ABC, abstractmethod


class IDemo(ABC):
    def __init__(self):
        self.running = False
        self.name = "Abstract Demo Class"

    @abstractmethod
    def start(self):
        pass

    def announceStart(self):
        print(f"Now running {self.name}...\n")

    def kill(self):
        print("Killing demo...")
        self.running = False
