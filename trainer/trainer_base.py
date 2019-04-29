from abc import ABCMeta, abstractmethod
class Agent_trainer(metaclass=ABCMeta):

    @abstractmethod
    def build_net(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def save_model(self):
        pass

    @abstractmethod
    def action(self):
        pass

    @abstractmethod
    def update_params(self):
        pass