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

    world = scenario.make_world()

    env = MultiAgentEnv(world,scenario.reset_world,scenario.reward,scenario.observation)

    return env,world


if __name__ == "__main__":
    # arglist = parse_args() TODO
    act_num = len(action_dict)
    env, world = make_env("trainer_1_test")

    trainer = T.DQN_trainer(env,world, n_actions=act_num)

    step  = 0

    for episode in range(TRAIN_STEP_MAX):
        observation = env.reset()
        for episode_step in range(episode_step_max):
            env.render()
            #time.sleep(0.5)
            action = trainer.choose_action(observation)
            action_env = np.reshape(np.array(action_dict[str(action)]), [1,5])
            observation_, reward, done, info = env.step(action_env)
            action = np.array([[action]])
            trainer.store_transition(observation, action, reward, observation_)  # 将当前观察,行为,奖励和下一个观察存储起来
            if episode_step % 50 == 0:
                trainer.learn()
            observation = observation_
        print(episode, " | game over")
        env.close()











