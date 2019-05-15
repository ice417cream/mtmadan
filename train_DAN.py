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

        print("act",j)


    def update(self,id,action,j):

        print("update", id)

class Net():
    def __init__(self):
        self.stauts_n = tf.placeholder(tf.float32, [None, obs_shape_n[0][0]], 'stauts-input')
        self.actions_n = tf.placeholder(tf.float32, [None, world.dim_p * 2 + 1], 'actions-input')
        self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

        mu, sigma, self.v = self.build_net()
        td = tf.subtract(self.v_target, self.v, name='TD_error')

        with tf.name_scope('c_loss'):
            self.c_loss = tf.reduce_mean(tf.square(td))

        self._action_n = tf.distributions.Normal(mu, sigma)

        with tf.name_scope('a_loss'):
            log_prob = self._action_n.log_prob(self.actions_n)
            exp_v = log_prob * tf.stop_gradient(td)
            entropy = self._action_n.entropy()  # encourage exploration
            self.exp_v = 0.01 * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)

        with tf.name_scope('update'):
            self.train_a = tf.train.GradientDescentOptimizer(0.001).minimize(self.a_loss)
            self.train_c = tf.train.GradientDescentOptimizer(0.001).minimize(self.c_loss)

    def build_net(self):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.stauts_n, 200, tf.nn.relu6, kernel_initializer=w_init, name='l_a')
            mu = tf.layers.dense(l_a, world.dim_p * 2 + 1, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, world.dim_p * 2 + 1, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.stauts_n, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')
        return mu, sigma, v

    def update(self, feed_dict):
        sess.run([self.train_a, self.train_c], feed_dict)


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
    '''
    stauts_n = tf.placeholder(tf.float32,[None,obs_shape_n[0][0]],'stauts-input')
    actions_n = tf.placeholder(tf.float32,[None,world.dim_p*2+1],'actions-input')
    v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

    w_init = tf.random_normal_initializer(0.,.1)
    with tf.variable_scope('actor'):
        l_a = tf.layers.dense(stauts_n,200,tf.nn.relu6,kernel_initializer=w_init,name='l_a')
        mu = tf.layers.dense(l_a,world.dim_p*2+1,tf.nn.tanh,kernel_initializer=w_init,name='mu')
        sigma = tf.layers.dense(l_a,world.dim_p*2+1,tf.nn.softplus,kernel_initializer=w_init,name='sigma')
    with tf.variable_scope('critic'):
        l_c = tf.layers.dense(stauts_n, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
        v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
    _action_n = tf.distributions.Normal(mu,sigma)
    td = tf.subtract(v_target, v, name='TD_error')
    #a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='/actor')
    with tf.name_scope('a_loss'):
        log_prob = _action_n.log_prob(actions_n)
        exp_v = log_prob * tf.stop_gradient(td)
        entropy = _action_n.entropy()  # encourage exploration
        exp_v = 0.01 * entropy + exp_v
        a_loss = tf.reduce_mean(exp_v)
        train_step = tf.train.AdamOptimizer(0.001).minimize(a_loss)
'''
    trainer = Net()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        _data = env.reset()
        data1 = np.ones([1000,5], dtype=np.float32)
        action_n  = []
        action_n.append(data1)
        
        for i in range(5000):

            data = {trainer.stauts_n: _data, trainer.actions_n: data1}
            time_start_action = time.time()
            #print("log_prob is ", sess.run(log_prob, feed_dict=data))
            #print("entropy is ", sess.run(entropy, feed_dict=data))
            #print("a_loss is ", sess.run(a_loss, feed_dict=data))
            action_n = (sess.run(trainer._action_n.sample(1), feed_dict=data))
            #sess.run(train_step, feed_dict=data)
            time_for_action = time.time() - time_start_action
            new_obs_n, rew_n, done_n, info_n = env.step(action_n[0])
            _data = new_obs_n
            data1 = action_n[-1]
            v_s_ = sess.run(trainer.v, {trainer.stauts_n: new_obs_n[np.newaxis, :]})[0,0]

            #更新网络
            trainer.update(feed_dict=data)

            print('='*500)
            print(time_for_action)
            env.render()
        print(action_n)

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









