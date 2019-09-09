"""
Centralized DDPG with multi-thread for actor update
"""
import tensorflow as tf
import numpy as np
from copy import copy
from memory import Memory, PrioritizedMemory
import matplotlib.pyplot as plt
import os


class DDPG:
    def __init__(self, dic_agent_conf, dic_exp_conf, dic_path):
        self.dic_agent_conf = dic_agent_conf
        self.dic_path = dic_path

        self.var = dic_agent_conf["ACTION_NOISE"]
        self.a_dim = dic_agent_conf["ACTION_DIM"]
        self.s_dim = dic_agent_conf["STATE_DIM"]
        self.memory_size = dic_agent_conf["MEMORY_SIZE"]
        self.batch_size = dic_agent_conf["BATCH_SIZE"]
        self.train = dic_agent_conf["TRAIN"]
        self.regularization = dic_agent_conf["REGULARIZATION"]

        self.target_replace_frequency = dic_agent_conf["TARGET_REPLACE_FREQ"]
        self.delay = dic_agent_conf["DELAY"]
        self.prob = dic_agent_conf["DELAY_PROB"]

        tau = dic_agent_conf["TARGET_REPLACE_RATIO"]
        lr_a = dic_agent_conf["ACTOR_LEARNING_RATE"]
        lr_c = dic_agent_conf["CRITIC_LEARNING_RATE"]
        gamma = dic_agent_conf["GAMMA"]

        self.train_counter = 0

        self.memory = Memory(self.memory_size, random_seed=self.dic_agent_conf["NUMPY_SEED"])


        tf.set_random_seed(dic_agent_conf["TENSORFLOW_SEED"])
        self.sess = tf.Session()

        self.S = tf.placeholder(tf.float32, [None, self.s_dim], 'state_input')
        self.h = tf.placeholder(tf.float32, [None, 32], 'rnn_hidden_state_placeholder')
        self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], 'next_state_input')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.rnn_hidden_state_input = np.zeros([1, 32])
        self.rnn_hidden_state_input_record = copy(self.rnn_hidden_state_input)
        with tf.variable_scope('Actor'):
            if self.dic_agent_conf["RNN"]:
                rnn_output, self.rnn_hidden_state = self.build_rnn(self.S, self.h)
                with tf.variable_scope("eval"):
                    self.a = self._build_a(rnn_output,  trainable=True)
                with tf.variable_scope("target"):
                    a_ = self._build_a(rnn_output, trainable=False)
            else:
                with tf.variable_scope("eval"):
                    self.a = self._build_a(self.S,  trainable=True)
                with tf.variable_scope("target"):
                    a_ = self._build_a(self.S_, trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        # target net replacement
        self.soft_replace = [[tf.assign(ta, (1 - tau) * ta + tau * ea), tf.assign(tc, (1 - tau) * tc + tau * ec)]
                             for ta, ea, tc, ec in zip(self.at_params, self.ae_params, self.ct_params, self.ce_params)]

        q_target = self.R + gamma * q_

        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.abs_errors = tf.abs(q_target - q)
        self.ctrain = tf.train.AdamOptimizer(lr_c).minimize(td_error, var_list=self.ce_params)

        if self.regularization:
            a_loss = - tf.reduce_mean(q) + tf.losses.get_regularization_loss()  # maximize the q
        else:
            a_loss = - tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(lr_a).minimize(a_loss, var_list=self.ae_params)

        self.saver = tf.train.Saver(max_to_keep=1)
        if self.train:
            self.sess.run(tf.global_variables_initializer())
        else:
            path = os.path.join(self.dic_path["DDPG"], str(self.prob), "") + dic_exp_conf["TIME"] + "/ckpt/"
            model_file = tf.train.latest_checkpoint(path)
            self.saver.restore(self.sess, model_file)

    def choose_action(self, s, explore=False):
        s_input = copy(s)

        if self.dic_agent_conf["RNN"]:
            self.rnn_hidden_state_input_record = copy(self.rnn_hidden_state_input)
            a, self.rnn_hidden_state_input = self.sess.run([self.a, self.rnn_hidden_state],
                                                           {self.S: s_input[np.newaxis, :],
                                                            self.h: self.rnn_hidden_state_input})
        else:
            a = self.sess.run(self.a, {self.S: s_input[np.newaxis, :]})

        a_output = a[0]

        if self.train or explore:
            if np.random.rand()<self.var:
                a_output=np.round(np.random.rand(),8)
            #a_output = np.clip(np.random.normal(a_output, self.var), 0, 1)
            self.var = self.var - 1 / self.memory_size if self.var > 0.001 else 0.001
            # a_output = np.clip(np.random.normal(a_output, self.var), 0, 1)
            # self.var = self.var - 1/self.memory_size if self.var > 0.001 else 0.001
        return a_output

    def convert_batch_to_input(self, batch):
        state, action, reward, next_state = [], [], [], [],
        if self.dic_agent_conf["RNN"]:
            hidden_state = []

        for data in batch:
            state.append(data["state"])
            action.append(data["action"])
            reward.append(data["reward"])
            next_state.append(data["next_state"])
            if self.dic_agent_conf["RNN"]:
                hidden_state.append(data["hidden_state"])

        state = np.vstack(state)
        action = np.vstack(action)
        reward = np.vstack(reward)
        next_state = np.vstack(next_state)
        if self.dic_agent_conf["RNN"]:
            hidden_state = np.vstack(hidden_state)
            return state, action, reward, next_state, hidden_state
        else:
            return state, action, reward, next_state

    def learn(self):
        # soft target replacement
        if self.train_counter % self.target_replace_frequency == 0:
            self.sess.run(self.soft_replace)

        batch, _ = self.memory.sample(self.batch_size)
        if self.dic_agent_conf["RNN"]:
            state, action, reward, next_state, hidden_state = self.convert_batch_to_input(batch)
        else:
            state, action, reward, next_state = self.convert_batch_to_input(batch)
        # if self.train_counter % self.memory_size == 0 and self.train_counter > self.memory_size//2:
        #    self._plot()
        if self.dic_agent_conf["RNN"]:
            self.sess.run(self.ctrain, {self.S: state, self.h: hidden_state, self.a: action, self.R: reward,
                                        self.S_: next_state})
            self.sess.run(self.atrain, {self.S: state, self.h: hidden_state})
        else:
            self.sess.run(self.ctrain, {self.S: state, self.a: action, self.R: reward, self.S_: next_state})
            self.sess.run(self.atrain, {self.S: state})
        self.train_counter += 1

    def store_transition(self, state, action, reward, next_state,):
        data = {"state": state, "action": action, "reward": reward, "next_state": next_state}
        if self.dic_agent_conf["RNN"]:
            data["hidden_state"] = self.rnn_hidden_state_input_record
        self.memory.store(data)

    def save_model(self, t):
        time = "{}_{}_{}_{}_{}".format(t[0], t[1], t[2], t[3], t[4], t[5])

        path = os.path.join(self.dic_path["DDPG"], str(self.prob), "") + time + "/ckpt"
        if not os.path.exists(path):
            os.makedirs(path)
        file_name = path + "/best" + ".ckpt"
        self.saver.save(self.sess, file_name)

    def load_best_model(self, t):
        tf.reset_default_graph()
        time = "{}_{}_{}_{}_{}".format(t[0], t[1], t[2], t[3], t[4], t[5])
        path = os.path.join(self.dic_path["DDPG"], str(self.prob), "") + time + "/ckpt/"
        model_file = tf.train.latest_checkpoint(path)
        self.saver.restore(self.sess, model_file)

    @staticmethod
    def build_rnn(s, h):
        rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=32, activation=tf.nn.tanh)
        output, h_next = rnn_cell.__call__(s, h)
        return output, h_next

    def _build_a(self, h, trainable,):
        units = self.dic_agent_conf["DENSE_UNITS"]
        regularizer = tf.contrib.layers.l2_regularizer(scale=self.dic_agent_conf["L2_REGULARIZATION"])
        if self.regularization:
            x = tf.layers.dense(h, units, trainable=trainable, kernel_regularizer=regularizer)
        else:
            x = tf.layers.dense(h, units, trainable=trainable)
        x = tf.nn.relu(x)

        if self.regularization:
            x = tf.layers.dense(x, units, trainable=trainable, kernel_regularizer=regularizer)
        else:
            x = tf.layers.dense(x, units, trainable=trainable)
        x = tf.nn.relu(x)

        x = tf.layers.dense(x, self.a_dim, trainable=trainable)
        x = tf.nn.sigmoid(x)

        return x

    def _build_c(self, s, a, scope, trainable):
        units = self.dic_agent_conf["DENSE_UNITS"]
        with tf.variable_scope(scope):
            x = tf.layers.dense(s, units, trainable=trainable)
            x = tf.nn.relu(x)

            a_ = tf.layers.dense(a, units, trainable=trainable)
            x = tf.concat([x, a_], axis=-1)

            x = tf.layers.dense(x, units, trainable=trainable)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), trainable=trainable)
        return x

    @property
    def memory_batch_full(self):
        if self.memory.memory_pointer >= self.batch_size:
            return True
        else:
            return False

    def _plot(self):
        batch_size = int(self.memory_size / 2)
        bt, _ = self.memory.sample(batch_size)
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        abs_error = self.sess.run(self.abs_errors, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
        abs_error = np.squeeze(abs_error)

        indices = self.memory.memory_pointer // self.memory_size
        plt.figure()
        plt.hist(abs_error)
        path = os.path.join(self.dic_path["DDPG"], self.dic_agent_conf["DELAY_PROB"], "误差")
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(path + "/" + str(indices) + ".png")


if __name__ == "__main__":
    from config import *
    from env import Env
    np.random.seed(dic_agent_conf["NUMPY_SEED"])
    dic_exp_conf["IMITATION"] = False

    env = Env(dic_env_conf)
    dic_exp_conf["UPPER_BOUND"] = env.upper_bound / env.normalizer

    if dic_exp_conf["IMITATION"]:
        dic_agent_conf["ACTION_NOISE"] *= 0.75

    dic_agent_conf["RNN"] = True
    agent = DDPG(dic_agent_conf, dic_exp_conf, dic_path)

    print("=== Build Agent: %s ===" % dic_exp_conf["AGENT_NAME"])

    # ===== train =====
    print("=== Train Start ===")
    train_reward = []
    train_rate = []
    train_q_size = []

    best_r = 0.0
    best_cnt_iter = None
    for cnt_train_iter in range(dic_exp_conf["TRAIN_ITERATIONS"]):
        s, other = env.reset()
        r_sum = 0
        for cnt_train_step in range(dic_exp_conf["MAX_EPISODE_LENGTH"]):

            a = agent.choose_action(s, explore=True)
            s_, r, done, info, other_ = env.step(a)
            r_sum += r
            train_reward.append(r_sum)
            train_rate.append(info[1])
            train_q_size.append(info[0])
            if "DDPG" in dic_exp_conf["AGENT_NAME"]:
                agent.store_transition(s, a, r, s_)

            s = s_
            other = other_
            if done:
                break

            if agent.memory_batch_full:
                agent.learn()
        print("train, iter:{}, r_sum:{}, info:{}".format(cnt_train_iter, r_sum, info))

    print("=== Train End ===")