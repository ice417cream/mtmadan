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
MAX_GLOBAL_EP = 2000
MAX_EP_STEP = 200
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.9
GLOBAL_RUNNING_R=[]
batch_size = 25

class Worker():
    def __init__(self,env, world, obs_shape_n, sess):
        print("Worker_init")
        self.trainer = T.mtmadan_trainer(env, world, obs_shape_n, sess)

    def act(self):
        print("act")
        _data = env.reset()
        buffer_s, buffer_a, buffer_r = [], [], []
        buffer_v_target = []
        ep_r = 0
        for i in range(batch_size):
            #time_start_action = time.time()
            action_n = self.trainer.action(_data)
            #time_for_action = time.time() - time_start_action
            new_obs_n, rew_n, done_n, info_n = env.step(action_n[0])

            #TODO 保存每一次step的输出

            _data = new_obs_n
            v_s_ = sess.run(self.trainer.v, {self.trainer.stauts_n: new_obs_n[np.newaxis, :]})[0,0]

            ep_r += rew_n
            buffer_s.append(new_obs_n)
            buffer_a.append(action_n)
            buffer_r.append((rew_n+8)/8)
            #print('='*500)
            #print(time_for_action)

            #显示训练结果图像
            env.render()

            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                buffer_v_target.append(v_s_)
        #收集后用于更新的数据
        feed_dict = {
            self.trainer.status_n: buffer_s,
            self.trainer.actions_n: buffer_a,
            self.trainer.v_target: buffer_v_target,
        }

        return feed_dict

    def update(self):
        print("update")

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
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    act_shape_n = [env.action_space[i] for i in range(env.n)] #返回值是离散空间Discrete

    with tf.Session() as sess:

        worker = Worker(env, world, obs_shape_n, sess)
        sess.run(tf.global_variables_initializer())
        feed_dict = worker.act()
        # 更新网络
        T.mtmadan_trainer.update_params(feed_dict)









