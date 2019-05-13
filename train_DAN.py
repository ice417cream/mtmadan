#liuchang @bit
import numpy as np
import tensorflow as tf
import time
import multiprocessing
import threading
import trainer.A3C_trainer as Trainer



#测试用模块
from gym import spaces


#初始参数 TODO
N_WORKERS = 15
OUT_GRAPH = True
ANYS_ONLINE = True
DISPLAY = True
cpu_number = multiprocessing.cpu_count()
GLOBAL_NET_SCOPE = 'Global_Net'
GLOBAL_EP = 0
MAX_GLOBAL_EP = 2000
MAX_EP_STEP = 200
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
GLOBAL_RUNNING_R=[]

class Worker():
    def __init__(self,name,env, world, obs_shape_n,SESS,GLOBAL_AC,NAME='sample',GLOBAL_NET_SCOPE = 'Global_Net'):
        print("Worker_init")

    def work(self):
        print("work")

#creat the world
def make_env(scenario_name):

    print("make_env")

    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    scenario = scenarios.load(scenario_name+".py").Scenario()

    world = scenario.make_world()

    env = MultiAgentEnv(world,scenario.reset_world,scenario.reward,scenario.observation)

    return env,world



if __name__=="__main__":

    # arglist = parse_args() TODO

    env, world= make_env("mtmadan_test")
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    act_shape_n = [env.action_space[i] for i in range(env.n)] #返回值是离散空间Discrete
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        workers = []
        GLOBAL_AC = Trainer.A3C_trainer('Global_Net',obs_shape_n[0][0],world.dim_p * 2 + 1,SESS)
        for i in range(N_WORKERS):#TODO
            i_name = 'W_%i' % i
            workers.append(Worker(i_name,env, world, obs_shape_n,SESS,GLOBAL_AC)) #TODO
        print("worker done")

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if DISPLAY:# TODO


        step = 0
        action_n = []
        print('DISPLAY')
        while True:
            env.reset()
            with tf.Session() as sess:
                for i in range(20):
                    for j in range(len(act_shape_n)):
                        action_n.append(tuple(np.random.rand(5)))
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


