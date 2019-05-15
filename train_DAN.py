#lewis @bit
import numpy as np
import tensorflow as tf
import time
import multiprocessing
import trainer.mtmadan_trainer as T
import threading

#初始参数 TODO
N_WORKERS = 15
OUT_GRAPH = True
ANYS_ONLINE = True
DISPLAY = True
cpu_number = multiprocessing.cpu_count()
GLOBAL_NET_SCOPE = 'Global_Net'
GLOBAL_EP = 0
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
GLOBAL_RUNNING_R=[]
batch_size = 25
TRAIN_STEP_MAX = 2000

class Worker():
    def __init__(self,env, world, sess):
        print("Worker_init")
        self.trainer = T.mtmadan_trainer(env, world, sess)

    def work(self,batch_size,TRAIN_STEP,type='display'):
        print("work")
        if  type=='display':
            while True:
                self.display(batch_size)
        elif type =='train':
            for TRAIN_STEP in range(TRAIN_STEP_MAX):
                self.act()

    def act(self,batch_size):
        pass

    def display(self,batch_size):
        _status = env.reset()
        for batch_step in range(batch_size):
            actions_n = self.trainer.action(_status)
            env.step(actions_n)
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

    worker = Worker(env, world)

    worker.work(batch_size, TRAIN_STEP, 'display')









