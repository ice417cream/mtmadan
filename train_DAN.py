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
    def __init__(self):
        print("Worker_init")

    def act(self,dic,j):
        c = dic[str(j)]
        print("act")
        for i in range(j):
            c.append(j)
        dic[str(j)]=c

    def update(self,id):
        print("update",id)




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
    global agent_id


    a = Worker()
    b = Worker()
    with multiprocessing.Manager() as manager:
        d = manager.dict()
        p_list = []
        for k in range(4):
            d[str(k)]=[0]

        for j in range(10):
            pool = multiprocessing.Pool(processes=4)
            for i in range(4):
                pool.apply_async(a.act, (d,i))
            pool.close()
            pool.join()
            print(d)







