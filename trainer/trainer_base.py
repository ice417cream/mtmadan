from abc import ABCMeta, abstractmethod
class Agent_trainer(metaclass=ABCMeta):

    @abstractmethod
    def build_net(self):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

    @abstractmethod
    def action(self, s):
        pass

    @abstractmethod
    def update_params(self, S, R, A, batch_size):
        pass