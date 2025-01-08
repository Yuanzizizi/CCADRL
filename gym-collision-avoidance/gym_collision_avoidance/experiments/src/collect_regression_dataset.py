import os
import numpy as np
import pickle
from tqdm import tqdm

os.environ['GYM_CONFIG_CLASS'] = 'CollectRegressionDataset'
from gym_collision_avoidance.envs import Config
import gym_collision_avoidance.envs.test_cases as tc
from gym_collision_avoidance.experiments.src.env_utils import run_episode, create_env

np.random.seed(0)

def _get_accumulate_reward(rewards):
    reward_sum = rewards[-1]
    n_rewards = len(rewards[:-1])

    for r_idx in reversed(range(0, n_rewards)):
        r = rewards[r_idx]
        reward_sum = Config.DISCOUNT * reward_sum + r
        rewards[r_idx] = reward_sum
    return rewards

def fill_motion(env, one_env, num_datapts):
    assert(Config.TRAIN_SINGLE_AGENT)
    obs = env.reset()
    STATES  = []  # np.empty((num_datapts, obs[0].shape[-1]-1))
    ACTIONS = []  # np.empty((num_datapts, 2))
    VALUES  = []  # np.empty((num_datapts, 1))
    ind = 0
    success_times = 0
    times = 0
    with tqdm(total=num_datapts) as pbar:
        while True:
            s  = []
            a = []
            r = []
            obs = env.reset()
            game_over = False

            while not game_over:
                action, _ = one_env.agents[0].policy.find_next_action_and_value(obs, one_env.agents, 0, one_env.map)
                s.append(obs[0, 1:])
                a.append(action)

                ind += 1
                pbar.update(1)

                obs, rewards, game_over, info = env.step([{}])
                r.append(rewards)
            if int(rewards[0]==Config.REWARD_AT_GOAL): success_times += 1
            times += 1
            STATES  += s
            ACTIONS += a
            VALUES  += _get_accumulate_reward(r)


            if ind >= num_datapts:
                # print('success rate ................... :', success_times/times)
                return np.array(STATES[:num_datapts]), np.array(ACTIONS[:num_datapts]), np.array(VALUES[:num_datapts])

                
def main():  
    filename_template = os.path.dirname(os.path.realpath(__file__)) + '/../../datasets/regression/{num_agents}_agents_{dataset_name}_cadrl_dataset_action_value_{mode}.p'
    # filename_template = '/checkpoints/datasets/regression/{num_agents}_agents_{dataset_name}_cadrl_dataset_action_value_{mode}.p'
    env, one_env = create_env()
    modes = [
        {
            'mode': 'train',
            'num_datapts': Config.TRAIN_NUM_DATAPTS,
        },
        {
            'mode': 'test',
            'num_datapts': Config.TEST_NUM_DATAPTS,
        },
    ]
    for mode in modes:
        STATES, ACTIONS, VALUES = fill_motion(env, one_env, num_datapts=mode['num_datapts'])

        filename = filename_template.format(mode=mode['mode'], dataset_name=Config.DATASET_NAME, num_agents=Config.MAX_NUM_AGENTS_IN_ENVIRONMENT)
        file_dir = os.path.dirname(filename)
        os.makedirs(file_dir, exist_ok=True)

        with open(filename, "wb") as f:
            pickle.dump([STATES,ACTIONS,VALUES], f, protocol=4)

    print("Files written.")

if __name__ == '__main__':
    main()


# A B C D E D C B A
