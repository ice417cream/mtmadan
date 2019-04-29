import tensorflow as tf
import numpy as np
import gym
from trainer.trainer_base import Agent_trainer

class A3C_trainer(Agent_trainer):
    def __init__(self,name, env, globalAC):
        print("mtmadan trianer init")
        self.env = env
        self.name = name
        self.AC = ACNet(name, globalAC)

    def train(self):
        print("training")
        global GLOBAL_RUNNING_R, GLOBAL_EP, MAX_GLOBAL_EP, SESS
        global MAX_EP_STEP,UPDATE_GLOBAL_ITER,GAMMA
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not self.coord.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()  # 列表
            ep_r = 0
            for ep_t in range(MAX_EP_STEP):
                # if self.name == 'W_0':
                #     self.env.render()
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)  # 列表
                done = True if ep_t == MAX_EP_STEP - 1 else False

                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)  # normalize

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    if done:
                        v_s_ = 0  # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:  # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
                        buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                        "golbal_len %d" % len(GLOBAL_RUNNING_R)
                    )
                    GLOBAL_EP += 1
                    break

    def build_net(self,scope):
        print("building net")
        w_init = tf.random_normal_initializer(0., .1)
        # 建立actor的网络
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')

        # 建立critic的网络
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, v, a_params, c_params

    def update_global(self, feed_dict):
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def save_model(self):
        print("saving model")

    def load_model(self):
        print("loading model")

    def action(self, s):
        print("action")
        s = s[np.newaxis, :]
        return SESS.run(self.A, {self.s: s})

    def update_params(self):
        print("update params")