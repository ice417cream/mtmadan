#lewis @bit
import numpy as np
import time
import argparse
import trainer.DQN_trainer as T

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

#参数解析
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--episode-step-max", type=int, default=25, help="maximum episode length")
    parser.add_argument("--train-step-max", type=int, default=4000, help="number of episodes")
    parser.add_argument("--agent-num", type=int, default=10, help="number of agent")
    parser.add_argument("--landmark-num", type=int, default=1, help="number of landmark")

    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=50, help="sample batch memory from all memory for training")
    parser.add_argument("--replace-target-iter", type=int, default=300, help="copy eval net params into target net")

    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="./save_model/model", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=100, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="/save_model/", help="directory in which training state and model are loaded")

    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--train", action="store_false", default=True)

    return parser.parse_args()

#creat the world
def make_env(arglist):

    print("make_env")
    from MAEnv.environment import MultiAgentEnv
    import MAEnv.scenarios as scenarios

    scenario = scenarios.load(arglist.scenario+".py").Scenario()#建立一个类
    world = scenario.make_World(arglist.agent_num, arglist.landmark_num)
    env = MultiAgentEnv(world,scenario.reset_world,scenario.reward,scenario.observation)

    return env,world


if __name__ == "__main__":
    arglist = parse_args()

    env, world = make_env(arglist)
    trainer = T.DQN_trainer(env, world, arglist, n_actions=len(action_dict))

    step  = 0

    if arglist.train:
        for episode in range(arglist.train_step_max):
            observation = env.reset()
            start = time.time()
            #随机生成agent索引
            agent_index = np.random.randint(0, arglist.agent_num)

            #前向n步，存储训练数据
            for episode_step in range(arglist.episode_step_max):
                action_env = []

                #选择动作
                action = trainer.choose_action(observation)

                for act in action:
                    action_env.append(action_dict[str(int(act))])#定义动作，采用字典的方式

                #执行一步动作 获取数据
                observation_, reward, done, info = env.step(action_env)
                # if episode_step == 0:
                #     reward_start = reward
                # elif episode_step == episode_step_max - 1:
                #     reward_end = reward
                action = np.reshape(action, [arglist.agent_num, 1])
                reward = np.reshape(reward, [arglist.agent_num, 1])
                trainer.store_transition(observation[agent_index],
                                         action[agent_index],
                                         reward[agent_index],
                                         observation_[agent_index])  # 将当前观察,行为,奖励和下一个观察存储起来
                observation = observation_
            # diff_rew = reward_start - reward_end
            # agent_index = np.argmax(diff_rew)

            #开始训练
            trainer.learn()
            end = time.time()
            if episode % arglist.save_rate == 0:
                trainer.save_model(arglist.save_dir, episode)
            print("Train_Step:", episode)
            print("cost:", trainer.cost_his[-1])
            # env.close()

    if arglist.display:
        while True:
            observation = env.reset()
            for i in range(50):
                start = time.time()
                action_env = []
                action = trainer.choose_action(observation)
                for act in action:
                    action_env.append(action_dict[str(int(act))])  # 定义动作，采用字典的方式
                observation_, reward, done, info = env.step(action_env)
                observation = observation_
                env.render()
                time.sleep(0.03)













