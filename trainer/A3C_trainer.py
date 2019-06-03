import tensorflow as tf
import numpy as np
import gym
from trainer.trainer_base import Agent_trainer

class A3C_trainer(Agent_trainer):
    def __init__(self, scope,N_S,N_A,SESS,globalAC = None):
        print("A3C_trianer init")
        self.SESS = SESS
        self.OPT_A = tf.train.RMSPropOptimizer(0.001, name='RMSPropA')
        self.OPT_C = tf.train.RMSPropOptimizer(0.01, name='RMSPropC')

        if scope == 'Global_Net':   # get global network
            print('global init')
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self.build_net(scope,N_A)[-2:]
                a = tf.constant(1)
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, N_A], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                mu, sigma, self.v, self.a_params, self.c_params = self.build_net(scope,N_A)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                # with tf.name_scope('wrap_a_out'):
                #     mu, sigma = mu * A_BOUND[1], sigma + 1e-4
                normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = normal_dist.entropy()  # encourage exploration
                    self.exp_v = 0.01 * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('choose_a'):  # use local params to choose action
                    self.A = mu#lc
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def build_net(self,scope,N_A):
        print("building net")
        w_init = tf.random_normal_initializer(0., .1)
        # 建立actor的网络
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 20000000, tf.nn.relu6, kernel_initializer=w_init, name='la')
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

        self.SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):  # run by a local

        self.SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def action(self,s):#/lc TODO
        ss = s[0]
        s = ss[np.newaxis, :]
        return self.SESS.run(self.A, {self.s: s})

    def update_params(self):
        pass

    def save_model(self):
        pass

    def load_model(self):
        pass