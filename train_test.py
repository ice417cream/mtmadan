#lewis @bit
import numpy as np
import tensorflow as tf
import time
import multiprocessing
import trainer.DQN_trainer as T
import threading
import os

#初始参数 TODO
batch_size = 50
TRAIN_STEP_MAX = 2000
save_path = "./save_model/model"
load_path = "./save_model/model-4"
load_model = False

#creat the world
def make_env(scenario_name):

    print("make_env")

    from MAEnv.environment import MultiAgentEnv
    import MAEnv.scenarios as scenarios

    scenario = scenarios.load(scenario_name+".py").Scenario()#建立一个类

    world = scenario.make_world()

    env = MultiAgentEnv(world,scenario.reset_world,scenario.reward,scenario.observation)

    return env,world


if __name__ == "__main__":
    # arglist = parse_args() TODO

    env, world = make_env("trainer_1_test")

    trainer = T.DQN_trainer(env,world)

    step  = 0

    for episode in range(300):
        observation = env.reset()
        while True:
            action = trainer.choose_action(observation)














