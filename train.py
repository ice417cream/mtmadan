import numpy as np
import tensorflow as tf
import time
import multiprocessing
import threading


#初始参数 TODO
N_WORKERS = 2
OUT_GRAPH = True
ANYS_ONLINE = True
DISPLAY = True

class Worker():
    def __init__(self):
        print("Worker_init")

    def work(self):

        print("work")

def make_env(scenario_name):

    print("make_env")
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario = scenarios.load(scenario_name+".py").Scenario()

    world = scenario.make_world()

    env = MultiAgentEnv(world,scenario.reset_world,scenario.reward,scenario.observation)

    return env


def get_trainers():
    print("make_env")


if __name__=="__main__":

    env = make_env("mtmadan_test")
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    act_shape_n = [env.action_space[i] for i in range(env.n)]#返回值是离散空间Discrete

    # arglist = parse_args() TODO
    SESS = tf.Session()
    with tf.device("/cpu:0"):
        workers = []
        for i in range(N_WORKERS):#TODO
            i_name = 'W_%i' % i
            workers.append(Worker()) #TODO

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if DISPLAY:# TODO
        print('DISPLAY')
        exit()

    worker_threads = []
    for worker in workers:
        job = lambda:worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    if ANYS_ONLINE:# TODO
        print('ANYS_ONLINE')


