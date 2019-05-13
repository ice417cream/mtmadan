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
DISPLAY = False
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
        print("get trainer")
        self.trainer = Trainer.A3C_trainer(name,obs_shape_n[0][0],world.dim_p * 2 + 1,SESS,GLOBAL_AC)#[0][0]is tuple-shape
        self.env = env
        self.name = name
    def work(self):
        print("work")
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()  # 重启世界，返回值是obs
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):  # 这里规定了更新步长  200
                a = self.trainer.action(s)#lc
                s_, r, done, info = self.env.step(a)  # 基于计算的随机数和边界选择一个作为动作,并输入到环境中去，用于获得新的动作、奖励等
                done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += np.sum(r)  # 注意这里并没有进行模型的更新，在模型的基础上进行了探索，以期待获得更好的回报可能
                buffer_s.append(s[0])
                buffer_a.append(a[0])
                buffer_r.append((np.sum(r) + 8) / 8)  # normalize 长期回报期望
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net  10次进行一次更新
                    # A3C_learning.py:132
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = SESS.run(self.trainer.v, {self.trainer.s: s_[0][np.newaxis, :]})[0, 0]  # 取运算结果的第一行第一列数,当前的动作得到的c
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r倒序读取数据
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)  # 定义
                    buffer_v_target.reverse()  # 重新再翻转一次
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)  # 将列表格式转换成矩阵，可以理解是行变列
                    feed_dict = {
                        self.trainer.s: buffer_s,
                        self.trainer.a_his: buffer_a,
                        self.trainer.v_target: buffer_v_target,
                    }
                    self.trainer.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []  # buffer只记录10次的内容
                    self.trainer.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    # print(
                    #     self.name,
                    #     "Ep:", GLOBAL_EP,
                    #     "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                    # )
                    GLOBAL_EP += 1
                    break

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


