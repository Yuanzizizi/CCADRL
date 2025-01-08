import os
import numpy as np
import gym
gym.logger.set_level(40)
os.environ['GYM_CONFIG_CLASS'] = 'Example'
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

from gym_collision_avoidance.envs.test_cases import *
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.util import *

map_list = {
    1:"length_map_41.jpg",
}

map_size_list = {
    1: (8, 8),
}

def main():
    import argparse
    parser = argparse.ArgumentParser(description='para transfer')
    parser.add_argument('--checkpt_name', type=str, help='checkpt_name -> str type.')
    parser.add_argument('--case_id', type=int, default=1, help='case_id -> int type.')
    parser.add_argument('--agent_number', type=int, default=5, help='para2 -> int type.')
    parser.add_argument('--mini', type=str, default=' ', help='mini -> str type.')
    parser.add_argument('--envname', type=str, default="get_testcase_two_agents", help='scenarios')


    args = parser.parse_args()
    print('=======================')
    print(args)
    print('=======================')

    if not args.mini == '':
        Config.SAVE_EPISODE_PLOTS = True
        Config.PLOT_CIRCLES_ALONG_TRAJ = True
        Config.ANIMATE_EPISODES = True

        args.mini = "-mini"

    base_path = os.path.dirname(os.path.abspath(__file__))  
    file_path = os.path.join(base_path, "/world_maps/"+map_list[args.case_id])

    Config.MAP_FILE = [file_path]
    Config.MAP_XW           = map_size_list[args.case_id][0]
    Config.MAP_YW           = map_size_list[args.case_id][1]
    Config.MAX_TIME_RATIO   = 4.

    '''
    Minimum working example:
    2 agents: 1 running external policy, 1 running GA3C-CADRL
    '''

    # Create single tf session for all experiments
    import tensorflow as tf

    
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.Session().__enter__()

    # Instantiate the environment
    env = gym.make("CollisionAvoidance-v0")

    # In case you want to save plots, choose the directory
    env.set_plot_save_dir(
        os.path.dirname(os.path.realpath(__file__)) + '/../../experiments/results/case' + str(args.case_id) + args.mini + '-' + str(args.agent_number) + '/')

    success_times = []
    collision_times = []
    timeout_times = []
    success = 0
    collision = 0
    wall = 0
    timeout = 0
    too_close = 0
    min_dist = [0]
    cumulative_rewards = []
    collision_cases = []
    timeout_cases = []
    average_lengths =[]
    fmm_base = []
    k= 1

    for i in range(k):
        agents = globals()[args.envname](i, args.case_id, args.agent_number, args.mini)
        base_path = os.path.dirname(os.path.abspath(__file__))  
        file_path = os.path.join(base_path, "checkpoints/")

        checkpt ={'checkpt_dir': file_path, 'checkpt_name': args.checkpt_name}
        [agent.policy.initialize_network(**checkpt) for agent in agents if hasattr(agent.policy, 'initialize_network')]
        env.set_agents(agents)

        obs = env.reset(test_case_index = i)
        done = False

        rewards_list = []
        rewards = [0]
        fmm_dists_base = agents[0].fmm_dist_to_goal  ###
        # lengths = []
        infos = {'which_agents_done':{0: False}}
        # while not (infos['which_agents_done'][0]==True):
        while not done:
            obs, rewards, done, infos, lengths = env.step({})
            rewards_list.append(rewards[0])

            if isinstance(infos['log_info'][0], Danger):
                too_close += 1
                min_dist.append(infos['log_info'][0].min_dist)
        env.reset(test_case_index = i)

        if isinstance(infos['log_info'][0], ReachGoal):
            success += 1
            success_times.append(agents[0].step_num)
            average_lengths.append(np.sum(lengths[0]))
            fmm_base.append(fmm_dists_base)
        elif isinstance(infos['log_info'][0], Collision):
            collision += 1
            collision_cases.append(i)
            collision_times.append(agents[0].step_num)
        elif isinstance(infos['log_info'][0], Wall):
            wall += 1
            collision_cases.append(i)
            collision_times.append(agents[0].step_num)
        else: 
            timeout += 1
            timeout_cases.append(i)
            timeout_times.append(agents[0].step_num)

        cumulative_rewards.append(_get_accumulate_reward(rewards_list)[0])

            
    success_rate = success / k
    collision_rate = collision / k
    wall_rate = wall / k
    assert success + collision + wall + timeout == k
    avg_nav_time = sum(success_times) / len(success_times) * Config.DT 
    average_lengths = np.mean(average_lengths)
    cumulative_rewards = np.mean(cumulative_rewards)
    avg_gmm_base = np.mean(fmm_base)
    extra_info = '' 
    print('{}has success rate: {:.2f}, collision with agent: {:.2f}, collision with wall: {:.2f}, nav time: {:.3f}({:.3f}), total reward: {:.4f}, nav length: {:.3f}({:.3f})'.
                    format(extra_info, success_rate, collision_rate, wall_rate, avg_nav_time-avg_gmm_base, avg_nav_time,
                        cumulative_rewards, average_lengths-avg_gmm_base, average_lengths))

    num_step = sum(success_times + collision_times + timeout_times) / Config.DT
    print('Frequency of being in danger: {:.2f} and average min separate distance in danger: {:.2f}'.format(too_close / num_step, average(min_dist)))

    print('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    print('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    return True

def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
        
def _get_accumulate_reward(rewards):  #######
    reward_sum = rewards[-1]
    n_rewards = len(rewards[:-1])

    for r_idx in reversed(range(0, n_rewards)):
        r = rewards[r_idx]
        reward_sum = Config.DISCOUNT * reward_sum + r
        rewards[r_idx] = reward_sum
    return rewards

if __name__ == '__main__':
    main()
    print("Experiment over.")


# A B C D E D C B A
