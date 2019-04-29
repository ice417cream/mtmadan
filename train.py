#liuchang @bit
import numpy as np
import tensorflow as tf
import time
import multiprocessing
import threading
import trainer.mtmadan_trainer as Trainer

#测试用模块
from gym import spaces


#初始参数 TODO
N_WORKERS = 2
OUT_GRAPH = True
ANYS_ONLINE = True

#A3C params, testing
MAX_GLOBAL_EP = 2000
MAX_EP_STEP = 200
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.0001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0


DISPLAY = True
cpu_number = multiprocessing.cpu_count()

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

    return env,world


def get_trainers(env,world,obs_shape_n,N_WORKERS=1):
    print("get_trainers")
    trainers = []
    trainer = Trainer.Mtmadan_trainer
    for i in range(N_WORKERS):
        trainers.append(trainer(env, world, obs_shape_n, i))
    return trainers


if __name__=="__main__":

    # arglist = parse_args() TODO

    env, world= make_env("mtmadan_test")
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    act_shape_n = [env.action_space[i] for i in range(env.n)] #返回值是离散空间Discrete
    trainers = get_trainers(env,world,obs_shape_n,N_WORKERS)


    SESS = tf.Session()
    with tf.device("/cpu:0"):
        workers = []
        for i in range(N_WORKERS):#TODO
            i_name = 'W_%i' % i
            workers.append(Worker()) #TODO

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if DISPLAY:# TODO

        random_a = tf.random_uniform([1,5], minval=0,maxval=1,dtype=tf.float32)#模型加载 TODO
        step = 0
        print('DISPLAY')
        while True:
            env.reset()
            with tf.Session() as sess:
                for i in range(20):
                    action_n = sess.run(random_a)
                    new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                    time.sleep(0.1)
                    env.render()
            step = step + 1
            print(step)

    worker_threads = []
    for worker in workers:
        job = lambda:worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

    if ANYS_ONLINE:# TODO
        print('ANYS_ONLINE')


