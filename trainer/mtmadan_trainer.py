from trainer.trainer_base import Agent_trainer
import numpy as np

class Mtmadan_trainer(Agent_trainer):
    def __init__(self, env, world, worker_count=1):
        print("mtmadan trianer init")

    def train(self):
        print("training")

    def build_net(self):
        print("building net")

    def save_model(self):
        print("saving model")

    def load_model(self):
        print("loading model")

    def action(self,_stauts):
        actions_n = np.random.rand(1000,5)
        return actions_n

    def update_params(self):
        print("update params")

