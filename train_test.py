#lewis @bit
import numpy as np
import tensorflow as tf
import time
import multiprocessing
import trainer.DQN_trainer as T
import threading
import os

#初始参数 TODO
batch_size = 50
TRAIN_STEP_MAX = 2000
episode_step_max = 100
save_path = "./save_model/model"
load_path = "./save_model/model-4"
load_model = False
agent_num = 100
landmark_num = 1
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
def make_env(scenario_name):

    print("make_env")

    from MAEnv.environment import MultiAgentEnv
    import MAEnv.scenarios as scenarios

    scenario = scenarios.load(scenario_name+".py").Scenario()#建立一个类

    world = scenario.make_World(agent_num, landmark_num)

    env = MultiAgentEnv(world,scenario.reset_world,scenario.reward,scenario.observation)

    return env,world


if __name__ == "__main__":
    # arglist = parse_args() TODO

    env, world = make_env("trainer_1_test")

    trainer = T.DQN_trainer(env,world, agent_num=agent_num)

    step  = 0

    for episode in range(TRAIN_STEP_MAX):
        observation = env.reset()
        start = time.time()

        agent_index = np.random.randint(0, agent_num)
        for episode_step in range(episode_step_max):
            env.render()
            #time.sleep(0.01)
            action_env = []
            action = trainer.choose_action(observation)
            for act in action:
                action_env.append(action_dict[str(int(act))])
            observation_, reward, done, info = env.step(action_env)
            # if episode_step == 0:
            #     reward_start = reward
            # elif episode_step == episode_step_max - 1:
            #     reward_end = reward
            action = np.reshape(action, [agent_num, 1])
            reward = np.reshape(reward, [agent_num, 1])
            trainer.store_transition(observation[agent_index],
                                     action[agent_index],
                                     reward[agent_index],
                                     observation_[agent_index])  # 将当前观察,行为,奖励和下一个观察存储起来
            observation = observation_
        # diff_rew = reward_start - reward_end
        # agent_index = np.argmax(diff_rew)
        trainer.learn()
        end = time.time()
        # if episode % 100 == 0:
        #     trainer.save_model(save_path, episode)
        print("Train_Step:", episode)
        env.close()











