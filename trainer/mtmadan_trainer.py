from trainer.trainer_base import Agent_trainer
import tensorflow as tf

class mtmadan_trainer(Agent_trainer):
    def __init__(self, env, world, sess,GAMMA=0.9):
        print("mtmadan trianer init")
        self.world = world
        self.sess = sess
        self.env = env

        self.obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        self.stauts_n = tf.placeholder(tf.float32, [None, self.obs_shape_n[0][0]], 'stauts-input')
        self.actions_n = tf.placeholder(tf.float32, [None, self.world.dim_p * 2 + 1], 'actions-input')
        self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
        self.GAMMA = GAMMA

        mu, sigma, self.v = self.build_net()
        td = tf.subtract(self.v_target, self.v, name='TD_error')

        with tf.name_scope('c_loss'):
            self.c_loss = tf.reduce_mean(tf.square(td))

        self._action_n = tf.distributions.Normal(mu, sigma)

        with tf.name_scope('a_loss'):
            log_prob = self._action_n.log_prob(self.actions_n)
            exp_v = log_prob * tf.stop_gradient(td)
            entropy = self._action_n.entropy()  # encourage exploration

            #0.01为可调参数
            self.exp_v = 0.01 * entropy + exp_v
            self.a_loss = tf.reduce_mean(-self.exp_v)

        with tf.name_scope('update'):
            self.train_a = tf.train.GradientDescentOptimizer(0.001).minimize(self.a_loss)
            self.train_c = tf.train.GradientDescentOptimizer(0.001).minimize(self.c_loss)

    def build_net(self):
        print("building net")
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.stauts_n, 200, tf.nn.relu6, kernel_initializer=w_init, name='l_a')
            mu = tf.layers.dense(l_a, self.world.dim_p * 2 + 1, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(l_a, self.world.dim_p * 2 + 1, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.stauts_n, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')
        return mu, sigma, v

    def save_model(self):
        print("saving model")

    def load_model(self):
        print("loading model")

    def action(self, s_n):
        print("action")
        action_n = self.sess.run(self._action_n.sample(1), feed_dict={self.stauts_n: s_n})
        return action_n

    def compute_global_r(self, obs_n_batch, reward_n_batch,GAMMA):
        print("compute_global_reward")
        buffer_v_target = []
        for r in reward_n_batch[::-1]:  # reverse buffer r倒序读取数据
            v_s_ = r + GAMMA * v_s_
            buffer_v_target.append(v_s_)  # 定义
        feed_dict = {}

        return feed_dict

    def update_params(self, obs_n_batch, reward_n_batch):
        print("update params")
        feed_dict = self.compute_global_r(obs_n_batch, reward_n_batch,self.GAMMA)
        self.sess.run([self.train_a, self.train_c], feed_dict)