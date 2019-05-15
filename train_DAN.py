#lewis @bit
import numpy as np
import tensorflow as tf
import time
import multiprocessing
import trainer.mtmadan_trainer as T
import threading
import os

#初始参数 TODO
batch_size = 3
TRAIN_STEP_MAX = 2000
save_path = "./save_model/model"
load_path = "./save_model/model-4"
load_model = False

class Worker():
    def __init__(self,env, world, sess):
        print("Worker_init")
        self.trainer = T.mtmadan_trainer(env, world, sess)

    def work(self,batch_size,TRAIN_STEP,type='display'):
        print("work")
        if type == 'display':
            while True:
                self.display(batch_size)
        elif type == 'train':
            for TRAIN_STEP in range(TRAIN_STEP_MAX):
                obs_n_batch, reward_n_batch, actions_n_batch = self.act(batch_size)
                self.trainer.update_params(obs_n_batch,reward_n_batch, actions_n_batch, batch_size)
                if TRAIN_STEP % 1 == 0:
                    if os.path.isdir(save_path) is False:
                        os.makedirs(save_path)
                    self.trainer.save_model(save_path, TRAIN_STEP)


    def act(self,batch_size):
        obs_n = env.reset()
        obs_n_batch = []
        reward_n_batch = []
        actions_n_batch = []
        for batch_step in range(batch_size):
            actions_n = self.trainer.action(obs_n)
            obs_n, reward_n, done_n, info_n = env.step(actions_n[0])
            obs_n_batch.append(obs_n)
            reward_n_batch.append(reward_n)
            actions_n_batch.append(actions_n[0])
            print("act", batch_step)
        return obs_n_batch, reward_n_batch, actions_n_batch

    def display(self,batch_size):
        _status = env.reset()
        for batch_step in range(batch_size):
            actions_n = self.trainer.action(_status)
            env.step(actions_n[0])
            print("act",batch_step)
            env.render()

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

    env, world = make_env("mtmadan_test")

    with tf.Session() as sess:

        worker = Worker(env, world, sess)
        sess.run(tf.global_variables_initializer())

        if load_model:
            worker.trainer.load_model(load_path)
            graph = tf.get_default_graph

        worker.work(batch_size, TRAIN_STEP_MAX, 'train')









