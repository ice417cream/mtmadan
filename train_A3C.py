#lewis @bit
import numpy as np
import time
import os
import shutil
import matplotlib.pyplot as plt
import threading
import tensorflow as tf
import trainer.A3C_trainer as T

OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = os.cpu_count()
MAX_EP_STEP = 5
MAX_GLOBAL_EP = 2000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
ENTROPY_BETA = 0.01
LR_A = 0.01    # learning rate for actor
LR_C = 0.01    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
agent_num = 1
landmark_num = 1
path = "./save_model/model"

#[stop, right, left, up, down]
action_dict = {"0": [0., 0., 1., 1., 0.], # l u
               "1": [0., 0., 0., 1., 0.], #   u
               "2": [0., 1., 0., 1., 0.], # r u
               "3": [0., 1., 0., 0., 0.], # r
               "4": [0., 1., 0., 0., 1.], # r d
               "5": [0., 0., 0., 0., 1.], #   d
               "6": [0., 0., 1., 0., 1.], # l d
               "7": [0., 0., 1., 0., 0.], # l
               "8": [1., 0., 0., 0., 0.]} #stop

#creat the world
def make_env(scenario):

    print("make_env")
    from MAEnv.environment import MultiAgentEnv
    import MAEnv.scenarios as scenarios

    scenario = scenarios.load(scenario+".py").Scenario()#建立一个类
    world = scenario.make_World(agent_num, landmark_num)
    env = MultiAgentEnv(world,scenario.reset_world,scenario.reward,scenario.observation)

    return env,world

class Worker(object):
    def __init__(self, name, env, globalAC):
        self.env = env
        self.name = name
        self.AC = T.ACNet(name, env, OPT_A, OPT_C, SESS, globalAC=globalAC)

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            agent_index = np.random.randint(0, agent_num)
            for ep_t in range(MAX_EP_STEP):
                action_env = []
                a = self.AC.choose_action(s)
                for act in a:
                    action_env.append(action_dict[str(int(act))])#定义动作，采用字典的方式
                s_, r, done, info = self.env.step(action_env)
                done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                buffer_s.append(s[agent_index, :])
                buffer_a.append(a[agent_index])
                buffer_r.append(r[0, agent_index])  # normalize
                buffer_s_.append(s_[agent_index, :])
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_})[0, 0]

                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()
                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)

                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: np.array(buffer_a),
                        self.AC.v_target: np.array(buffer_v_target),
                    }
                    print(ep_t, " | a c loss", SESS.run([self.AC.a_loss, self.AC.c_loss], feed_dict=feed_dict))
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r, buffer_s_ = [], [], [], []
                    self.AC.pull_global()
                    agent_index = np.random.randint(0, agent_num)
                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    # print(self.name,
                    #     " | Ep:", GLOBAL_EP,
                    #     " | Ep_r:", GLOBAL_RUNNING_R[-1],
                    #     " | golbal_len:", len(GLOBAL_RUNNING_R))
                    GLOBAL_EP += 1
                    break

    def display(self):
            for i in range (10):
                observation = env.reset()
                for i in range(50):
                    action_env = []
                    action = self.AC.choose_action(observation)
                    for act in action:
                        action_env.append(action_dict[str(int(act))])  # 定义动作，采用字典的方式
                    observation_, reward, done, info = env.step(action_env)
                    observation = observation_
                    env.render()
                    time.sleep(0.03)

def save_model(path):
    if os.path.isdir(path) is False:
        os.makedirs(path)
    saver.save(SESS, save_path=path)

def load_model(path):
    print("loading model")
    load = tf.train.latest_checkpoint(path)
    saver.restore(SESS, load)

if __name__ == "__main__":

    SESS = tf.Session()
    env, world = make_env("trainer_1_test")
    with tf.device("/cpu:0"):
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = T.ACNet(GLOBAL_NET_SCOPE, env, OPT_A, OPT_C, SESS)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, env, GLOBAL_AC))
        worker_d = Worker("display", env, GLOBAL_AC)

    #加入线程协调器
    COORD = tf.train.Coordinator()

    #初始化所有变量
    SESS.run(tf.global_variables_initializer())
    saver = tf.train.Saver()

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            #递归删除文件夹
            shutil.rmtree(LOG_DIR)
        #将tf的图写入文件夹
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    save_model(path)
    worker_d.display()

    # plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    # plt.xlabel('step')
    # plt.ylabel('Total moving reward')
    # plt.show()