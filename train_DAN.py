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
batch_size = 25
TRAIN_STEP = 1000

class Worker():
    def __init__(self,env,world,trainer='MADAN'):
        print("Worker_init")
        self.trainer = Trainer.Mtmadan_trainer(env,world)

    def work(self,batch_size,TRAIN_STEP,type='display'):
        print("work")
        if  type=='display':
            while True:
                self.act(batch_size)



    def act(self,batch_size):
        _status = env.reset()
        batch_obs_n = []
        for batch_step in range(batch_size):
            actions_n = self.trainer.action(_status)
            obs_n = env.step(actions_n)
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



if __name__=="__main__":

    # arglist = parse_args() TODO

    env, world= make_env("mtmadan_test")

    worker = Worker(env,world)

    worker.work(batch_size,TRAIN_STEP,'display')


    # obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    # act_shape_n = [env.action_space[i] for i in range(env.n)] #返回值是离散空间Discrete
    #
    # stauts_n = tf.placeholder(tf.float32,[None,obs_shape_n[0][0]],'stauts-input')
    # actions_n = tf.placeholder(tf.float32,[None,world.dim_p*2-1],'actions-input')
    #
    # w_init = tf.random_normal_initializer(0.,.1)
    # with tf.variable_scope('actor'):
    #     l_a = tf.layers.dense(stauts_n,32,tf.nn.relu6,kernel_initializer=w_init,name='l_a')
    #     mu = tf.layers.dense(l_a,world.dim_p*2+1,tf.nn.tanh,kernel_initializer=w_init,name='mu')
    #     sigma = tf.layers.dense(l_a,world.dim_p*2+1,tf.nn.softplus,kernel_initializer=w_init,name='sigma')
    # _action_n = tf.distributions.Normal(mu,sigma).sample(1)
    #
    #
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     _data = env.reset()
    #     for i in range(50):
    #         data = {stauts_n: _data}
    #         time_start_action = time.time()
    #         action_n = sess.run(_action_n, feed_dict=data)
    #         time_for_action = time.time() - time_start_action
    #         new_obs_n, rew_n, done_n, info_n = env.step(action_n[0])
    #         _data = new_obs_n
    #         print('='*500)
    #         print(time_for_action)
    #         env.render()
    #     print(action_n)

    # 测试动作的
    # action_nn=[]
    # for i in range(1000):
    #     action_nn.append([0,1,0,1,0])
    # print(action_nn)



    # 随机生成动作指令，大约e-5的量级，可以承受
    # time_start_action = time.time()
    # action_n = np.random.rand(1000,5)
    # time_for_action = time.time()-time_start_action




    # global agent_id
    #
    #
    # a = Worker()
    # b = Worker()
    # with multiprocessing.Manager() as manager:
    #     d = manager.dict()
    #     e = multiprocessing.Event()
    #     action = manager.dict()
    #     p_list = []
    #
    #     for i in range(4):
    #         d[str(i)] = 0
    #     for i in range(10):
    #         action[str(i)] = 0
    #
    #     for j in range(10):
    #         pool = multiprocessing.Pool(processes=5)
    #         for i in range(4):
    #             pool.apply_async(a.act, (d,i))
    #         pool.apply(a.update, (5,action,j))
    #     pool.close()
    #     pool.join()
    #     print(action)









