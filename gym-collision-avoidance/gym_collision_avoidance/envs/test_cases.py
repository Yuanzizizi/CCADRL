import numpy as np
from numpy.linalg import norm

from gym_collision_avoidance.envs.agent import Agent

# Policies
from gym_collision_avoidance.envs.policies.StaticPolicy import StaticPolicy
from gym_collision_avoidance.envs.policies.NonCooperativePolicy import NonCooperativePolicy_v1, NonCooperativePolicy_v2
# from gym_collision_avoidance.envs.policies.DRLLongPolicy import DRLLongPolicy
# from gym_collision_avoidance.envs.policies.SARLPolicy import SARLPolicy
# from gym_collision_avoidance.envs.policies.RVOPolicy import RVOPolicy
# from gym_collision_avoidance.envs.policies.CADRLPolicy import CADRLPolicy
from gym_collision_avoidance.envs.policies.GA3CCADRLPolicy import GA3CCADRLPolicy
# from gym_collision_avoidance.envs.policies.ExternalPolicy import ExternalPolicy
# from gym_collision_avoidance.envs.policies.LearningPolicy import LearningPolicy
# from gym_collision_avoidance.envs.policies.CARRLPolicy import CARRLPolicy
from gym_collision_avoidance.envs.policies.LearningPolicyGA3C import LearningPolicyGA3C

# Dynamics
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics

# Sensors
# from gym_collision_avoidance.envs.sensors.OccupancyGridSensor import OccupancyGridSensor
from gym_collision_avoidance.envs.sensors.LaserScanSensor import LaserScanSensor
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor
# from gym_collision_avoidance.envs.sensors.Obstacle_rays import ObstacleRaysSensor ##
# from gym_collision_avoidance.envs.sensors.Distance_rays import DistanceRaysSensor ##

from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs.Map import Map

import random

# from gym_collision_avoidance.envs.policies.CADRL.scripts.multi import gen_rand_testcases as tc

test_case_filename = "{dir}/test_cases/{pref_speed_string}{num_agents}_agents_{num_test_cases}_cases.p"

policy_dict = {
    # 'RVO': RVOPolicy,
    'noncoopv1': NonCooperativePolicy_v1,
    'noncoopv2': NonCooperativePolicy_v2,
    # 'external': ExternalPolicy,
    'GA3C_CADRL': GA3CCADRLPolicy,
    # 'learning': LearningPolicy,
    'learning_ga3c': LearningPolicyGA3C,
    'static': StaticPolicy,
    # 'CADRL': CADRLPolicy,
    # 'SARL': SARLPolicy
}

sensor_dict = {
    'other_agents_states': OtherAgentsStatesSensor,
    'laserscan': LaserScanSensor,
    # 'occupancy_grid': OccupancyGridSensor,
    # 'obstacle_rays': DistanceRaysSensor, #ObstacleRaysSensor,
    # 'distance_rays': DistanceRaysSensor 
    # 'other_agents_states_encoded': OtherAgentsStatesSensorEncode,
}

dynamics_dict = {
    'unicycle': UnicycleDynamics,
}

# def gen_circle_test_case(num_agents, radius):
#     tc = np.zeros((num_agents, 6))
#     for i in range(num_agents):
#         tc[i, 4] = 1.0
#         tc[i, 5] = 0.5
#         theta_start = (2*np.pi/num_agents)*i
#         theta_end = theta_start + np.pi
#         tc[i, 0] = radius*np.cos(theta_start)
#         tc[i, 1] = radius*np.sin(theta_start)
#         tc[i, 2] = radius*np.cos(theta_end)
#         tc[i, 3] = radius*np.sin(theta_end)
#     return tc


def get_testcase_two_agents(i, case_id, agent_number, mini, speed_bnds=Config.TEST_CASE_ARGS['speed_bnds'], radius_bnds=Config.TEST_CASE_ARGS['radius_bnds']):
    agent_number =agent_number#np.random.randint(2, 11)
    
    x_width = Config.MAP_XW # meters
    y_width = Config.MAP_YW # meters
    grid_cell_size = Config.GRID_CELL_SIZE # meters/grid cell
    static_map = Map(x_width, y_width, grid_cell_size, Config.MAP_FILE[0])

    cadrl_test_case = np.array(
# [0.33774115804478644, -3.250628260194837, -1.5189736946542958, 3.408908344187766, 1.0, 0.3, 1.9670172826568422, 3.00951763953535, -1.6448702990118118, -3.2233208941112794, 1.0, 0.3, 3.0746405019128726, 3.520452679532344, -2.828287690921621, -3.336041042753597, 1.0, 0.3, 0.6085062275098627, 3.1625918447209047, 2.283869074261049, -3.2890396265845347, 1.0, 0.3, 1.6910372825592974, -3.489926253874547, -3.4538013819260813, 3.0337515860203403, 1.0, 0.3]
# [0.0799442819857914, -3.2954800233956165, -0.10114733688546096, 3.158151669483791, 1.0, 0.3, 3.0310196336068866, 3.064843860470039, 0.952577481821633, -3.467049000599256, 1.0, 0.3, 1.1464674074831827, -3.0532287530456728, 0.9321247073307966, 3.3531189274073974, 1.0, 0.3, -2.017523549557702, 3.5499624798987734, -2.739849611652863, -3.006046687400417, 1.0, 0.3, -0.7419485939683588, 3.4143065656884306, -1.2667858003423955, -3.312968073837499, 1.0, 0.3]
# [0.46767771895382637, -3.4406911489893153, -3.2113674870109934, 3.559132895546633, 1.0, 0.3, -3.309001513287991, -3.4809475758644792, -2.4570161316377837, 3.207022300254604, 1.0, 0.3, 1.5720635356540713, -3.4669336756453304, -1.4660324518984575, 3.490321033300045, 1.0, 0.3, 2.3174529353312954, 3.064030298845787, 2.018899188813835, -3.232829637675148, 1.0, 0.3, 3.4560273898086837, -3.373785801508389, 1.3260448480384657, 3.1755365823471973, 1.0, 0.3]
# [-1.66330277827808, -3.171294066276654, 0.3421758489484157, 3.2856090130360958, 1.0, 0.3, 2.228661398646601, 3.0247586111306752, -2.7666188599721835, -3.290886602915679, 1.0, 0.3, 2.992057155338192, 3.407086503383372, 1.5561921446369595, -3.2629382937582454, 1.0, 0.3, -0.4187084208504954, -3.240521480482383, 1.3343573511763074, 3.2620729126212638, 1.0, 0.3, -2.8392259397214827, -3.4726460334067406, -1.4972739983343653, 3.3322973480117155, 1.0, 0.3]
[2.398128106004253, 3.4151907700532105, 1.1389019274631451, -3.4674897518658505, 1.0, 0.3, -0.6132286017765312, 3.25396367328531, -2.9412181594990763, -3.491216849419675, 1.0, 0.3, -0.9266150925896426, -3.0269769680048055, -1.8178375031130338, 3.009091511380883, 1.0, 0.3, 2.05002324175795, -3.5321141266397342, 2.2068846940314417, 3.2356129139653804, 1.0, 0.3, -2.881317937548853, 3.299348285643725, -0.056730819028482316, -3.5328130672285605, 1.0, 0.3]
    )
    cadrl_test_case = cadrl_test_case.reshape(agent_number, 6)

    agents = cadrl_test_case_to_agents(cadrl_test_case,
            policies=  'GA3C_CADRL',
            policy_distr=None,
            agents_dynamics="unicycle",
            agents_sensors=Config.TEST_CASE_ARGS['agents_sensors'],
            policy_to_ensure=None,
            prev_agents=None,
            static_map=static_map, 
            )

    return agents

# def get_testcase_random(num_agents=None, side_length=4, speed_bnds=[0.5, 2.0], radius_bnds=[0.2, 0.8], policies='learning', policy_distr=None, agents_dynamics='unicycle', agents_sensors=['other_agents_states'], policy_to_ensure=None, prev_agents=None):
#     if num_agents is None:
#         num_agents = np.random.randint(2, Config.MAX_NUM_AGENTS_IN_ENVIRONMENT+1)

#     # if side_length is a scalar, just use that directly (no randomness!)
#     if type(side_length) is list:
#         # side_length lists (range of num_agents, range of side_lengths) dicts
#         # to enable larger worlds for larger nums of agents (to somewhat maintain density)
#         for comp in side_length:
#             if comp['num_agents'][0] <= num_agents < comp['num_agents'][1]:
#                 side_length = np.random.uniform(comp['side_length'][0], comp['side_length'][1]) 
#         assert(type(side_length) == float)

#     cadrl_test_case = tc.generate_rand_test_case_multi(num_agents, side_length, speed_bnds, radius_bnds)

#     agents = cadrl_test_case_to_agents(cadrl_test_case,
#         policies=policies,
#         policy_distr=policy_distr,
#         agents_dynamics=agents_dynamics,
#         agents_sensors=agents_sensors,
#         policy_to_ensure=policy_to_ensure,
#         prev_agents=prev_agents
#         )
#     return agents


def get_testcase_map_random(speed_bnds, radius_bnds, policy_to_ensure, policies, policy_distr, agents_sensors, 
                            prev_agents=None, agents_dynamics='unicycle', static_map=None, num_agents=None):
    pre_agents_num = 12 if Config.USE_PLAN2 else 0
    if num_agents is None:
        num_agents = np.random.randint(2, Config.MAX_NUM_AGENTS_IN_ENVIRONMENT-pre_agents_num+1) if Config.MAX_NUM_AGENTS_IN_ENVIRONMENT > 1 else 1
    if '0' in static_map.map_filename and np.random.rand() < 0.9:
        if np.random.rand() < 0.5:
            cadrl_test_case = generate_circle_case(num_agents=num_agents, speed_bnds=speed_bnds, radius_bnds=radius_bnds)  
        else:
            cadrl_test_case = generate_swap_case(num_agents=num_agents, speed_bnds=speed_bnds, radius_bnds=radius_bnds)  
    elif 'narrow_gate' in static_map.map_filename:
            cadrl_test_case = generate_narrow_gate_case(num_agents=num_agents, speed_bnds=speed_bnds, radius_bnds=radius_bnds, static_map=static_map)  
    else:	
        cadrl_test_case = generate_randposs_for_mapEnv(speed_bnds=speed_bnds, radius_bnds=radius_bnds, static_map=static_map, num_agents=num_agents)
    
    agents = cadrl_test_case_to_agents(cadrl_test_case,
		policies=policies,
		policy_distr=policy_distr,
		agents_dynamics=agents_dynamics,
		agents_sensors=agents_sensors,
		policy_to_ensure=policy_to_ensure,
		prev_agents=prev_agents,
		static_map=static_map)

    return agents

# def generate_randposs_for_mapEnv(speed_bnds, radius_bnds, static_map, num_agents):
#     # px py gx gy speed radius
#     test_case = np.zeros((num_agents, 6))
#     for i in range(num_agents):
# 		# speed and radius
#         test_case[i,4] = (speed_bnds[1]  - speed_bnds[0])  * np.random.rand() + speed_bnds[0]
#         test_case[i,5] = (radius_bnds[1] - radius_bnds[0]) * np.random.rand() + radius_bnds[0]
        
#         while True:           
#             # generate random starting/ending points
#             start = np.array([Config.MAP_XW, Config.MAP_YW]) * (np.random.rand(2,)-0.5) 
#             end   = np.array([Config.MAP_XW, Config.MAP_YW]) * (np.random.rand(2,)-0.5) 

#             if static_map.check_feasible(start) or static_map.check_feasible(end):
#                 if_collide = True
#                 continue
                
#             # check shortest path length
#             if np.linalg.norm(start-end) < np.mean([Config.MAP_XW, Config.MAP_YW]) * 0.5:
#                 continue

#             # if colliding with previous test cases
#             if_collide = False
#             for j in range(i):
#                 dis_lim = test_case[j,5] + test_case[i,5] + Config.GETTING_CLOSE_RANGE
#                 if np.linalg.norm(start - test_case[j,0:2]) < dis_lim or \
#                    np.linalg.norm(end   - test_case[j,2:4]) < dis_lim:
#                    if_collide = True
#                    break
#             if if_collide:
#                 continue

#             break

#         test_case[i,0:2] = start
#         test_case[i,2:4] = end

#     return test_case

# def generate_randposs_for_mapEnv(speed_bnds, radius_bnds, static_map, num_agents):
#     # px py gx gy speed radius
#     test_case = np.zeros((num_agents, 6))
#     for i in range(num_agents):
# 		# speed and radius
#         test_case[i,4] = (speed_bnds[1]  - speed_bnds[0])  * np.random.rand() + speed_bnds[0]
#         test_case[i,5] = (radius_bnds[1] - radius_bnds[0]) * np.random.rand() + radius_bnds[0]
        
#         while True:           
#             # generate random starting/ending points

#             # random.seed(5)
#             sign_y = -1 if np.random.random() < 0.5 else 1  # -1 上到下，+1 下到上
#             start = np.array([np.random.uniform(1.5, 3.5) * (-1 if np.random.random() < 0.5 else 1),
#                               np.random.uniform(1.5, 3.5) * (+sign_y), ])
#             end   = np.array([np.random.uniform(1.5, 3.5) * (-1 if np.random.random() < 0.5 else 1),
#                               np.random.uniform(1.5, 3.5) * (-sign_y)])

           

#             if static_map.check_feasible(start) or static_map.check_feasible(end):
#                 if_collide = True
#                 continue
                
#             # # check shortest path length
#             # if np.linalg.norm(start-end) < np.mean([Config.MAP_XW, Config.MAP_YW]) * 0.5:
#             #     continue

#             # if colliding with previous test cases
#             if_collide = False
#             for j in range(i):
#                 dis_lim = test_case[j,5] + test_case[i,5] + Config.GETTING_CLOSE_RANGE
#                 #    np.linalg.norm(end   - test_case[j,0:2]) < dis_lim or \
#                 #    np.linalg.norm(start - test_case[j,2:4]) < dis_lim or \    
#                 if np.linalg.norm(start - test_case[j,0:2]) < dis_lim or \
#                    np.linalg.norm(end   - test_case[j,2:4]) < dis_lim or \
#                    np.linalg.norm(end   - test_case[j,0:2]) < dis_lim or \
#                    np.linalg.norm(start - test_case[j,2:4]) < dis_lim:
#                    if_collide = True
#                    break
#             if if_collide:
#                 continue

#             break

#         test_case[i,0:2] = start
#         test_case[i,2:4] = end
#     return test_case


# def generate_randposs_for_mapEnv(speed_bnds, radius_bnds, static_map, num_agents):
#     # px py gx gy speed radius
#     test_case = np.zeros((num_agents, 6))
#     for i in range(num_agents):
# 		# speed and radius
#         test_case[i,4] = (speed_bnds[1]  - speed_bnds[0])  * np.random.rand() + speed_bnds[0]
#         test_case[i,5] = (radius_bnds[1] - radius_bnds[0]) * np.random.rand() + radius_bnds[0]
        
#         while True:           
#             # generate random starting/ending points

#             # random.seed(5)
#             sign = 1 if np.random.random() < 0.5 else -1

#             start_y =  random.uniform(20.0, 35.0) * 0.1
#             start_x = -random.uniform(-35.0, 35.0) * 0.1
#             if start_x < 0.75 and start_x > -0.75:
#                   continue
#             start = np.array([start_x, sign*start_y])

#             start_y =  random.uniform(20.0, 35.0) * 0.1
#             start_x = -random.uniform(-35.0, 35.0) * 0.1
#             if start_x < 0.75 and start_x > -0.75:
#                   continue
#             end = np.array([start_x, -sign*start_y])


#             if static_map.check_feasible(start) or static_map.check_feasible(end):
#                 if_collide = True
#                 continue
                
#             # # check shortest path length
#             # if np.linalg.norm(start-end) < np.mean([Config.MAP_XW, Config.MAP_YW]) * 0.5:
#             #     continue

#             # if colliding with previous test cases
#             if_collide = False
#             for j in range(i):
#                 dis_lim = test_case[j,5] + test_case[i,5] + Config.GETTING_CLOSE_RANGE
#                 #    np.linalg.norm(end   - test_case[j,0:2]) < dis_lim or \
#                 #    np.linalg.norm(start - test_case[j,2:4]) < dis_lim or \    
#                 if np.linalg.norm(start - test_case[j,0:2]) < dis_lim or \
#                    np.linalg.norm(end   - test_case[j,2:4]) < dis_lim or \
#                    np.linalg.norm(end   - test_case[j,0:2]) < dis_lim or \
#                    np.linalg.norm(start - test_case[j,2:4]) < dis_lim:
#                    if_collide = True
#                    break
#             if if_collide:
#                 continue

#             break

#         test_case[i,0:2] = start
#         test_case[i,2:4] = end
#     return test_case

def generate_circle_case(num_agents, speed_bnds, radius_bnds):
	r_min = num_agents / 2.0
	r = np.random.rand() * 2.0 + r_min
	r = min(r, Config.MAP_XW/2-1)
	
	test_case = np.zeros((num_agents, 6))
	counter = 0
	for i in range(num_agents):
		counter = 0
		test_case[i,5] = (radius_bnds[1] - radius_bnds[0]) * np.random.rand() + radius_bnds[0]
		test_case[i,4] = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]

		while True:
			if counter > 10:
				r *= 1.01
				r =  min(r, Config.MAP_XW/2-1)
				counter = 0
			start_angle = np.random.rand() * 2 * np.pi - np.pi  
			end_angle = np.pi + start_angle
			start = np.array([r*np.cos(start_angle), r*np.sin(start_angle)])
			end = np.array([r*np.cos(end_angle), r*np.sin(end_angle)])
			if_collide = False
			for j in range(i):
				radius_start = test_case[j,5] + test_case[i,5] + Config.GETTING_CLOSE_RANGE
				radius_end   = test_case[j,5] + test_case[i,5] + Config.GETTING_CLOSE_RANGE
				if np.linalg.norm(start - test_case[j,0:2] ) < radius_start or np.linalg.norm(end - test_case[j,2:4]) < radius_end:
					if_collide = True
					break

			if if_collide:
				counter += 1
				continue

			break

		test_case[i,0:2] = start
		test_case[i,2:4] = end
	return test_case


def generate_swap_case(num_agents, speed_bnds, radius_bnds):
	r_min = num_agents / 2.0
	r = np.random.rand() * 2.0 + r_min
	r = min(r, Config.MAP_XW/2-1)
	
	test_case = np.zeros((num_agents, 6))
	counter = 0
	for i in range(num_agents):
		counter = 0
		test_case[i,5] = (radius_bnds[1] - radius_bnds[0]) * np.random.rand() + radius_bnds[0]
		test_case[i,4] = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]

		while True:
			if counter > 10:
				r *= 1.01
				r =  min(r, Config.MAP_XW/2-1)
				counter = 0
			sign = 1 if np.random.rand() < 0.5 else -1
			start = np.array([ sign*r, (np.random.rand() * 2 - 1)  * r]) 
			end   = np.array([-sign*r, (np.random.rand() * 2 - 1)  * r]) 

			if_collide = False
			for j in range(i):
				radius_start = test_case[j,5] + test_case[i,5] + Config.GETTING_CLOSE_RANGE
				radius_end   = test_case[j,5] + test_case[i,5] + Config.GETTING_CLOSE_RANGE
				if np.linalg.norm(start - test_case[j,0:2] ) < radius_start or np.linalg.norm(end - test_case[j,2:4]) < radius_end:
					if_collide = True
					break

			if if_collide == True:
				counter += 1
				continue

			break

		test_case[i,0:2] = start
		test_case[i,2:4] = end
	return test_case


def generate_narrow_gatetest_case(num_agents, speed_bnds, radius_bnds, static_map):
	test_case = np.zeros((num_agents, 6))
	for i in range(num_agents):
		test_case[i,5] = (radius_bnds[1] - radius_bnds[0]) * np.random.rand() + radius_bnds[0]
		test_case[i,4] = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]

		while True:
			y 		= (Config.MAP_XW/2 - 0.6) * (np.random.rand() * 2 - 1)
			x_start = (Config.MAP_XW/2 - 0.6) * (np.random.rand() * 2 - 1)
			x_end   = (Config.MAP_XW/2 - 0.6) * (np.random.rand() * 2 - 1)

			if abs(x_end) < 1 and abs(y) < 2:
				continue

			sign = 1 if np.random.rand() < 0.5 else -1
			start = np.array([x_start, sign*y]) 
			end   = np.array([x_end,  -sign*y])
			
			if static_map.check_feasible(start) or static_map.check_feasible(end):
				continue
			
			if_collide = False
			for j in range(i):
				radius_start = test_case[j,5] + test_case[i,5] + Config.GETTING_CLOSE_RANGE
				radius_end   = test_case[j,5] + test_case[i,5] + Config.GETTING_CLOSE_RANGE
				if np.linalg.norm(start - test_case[j,0:2] ) < radius_start or np.linalg.norm(end - test_case[j,2:4]) < radius_end:
					if_collide = True
					break
					
			if if_collide:
				continue
			
			break

		test_case[i,0:2] = start
		test_case[i,2:4] = end
	return test_case


def generate_narrow_gate_case(num_agents, speed_bnds, radius_bnds, static_map):
	test_case = np.zeros((num_agents, 6))
	for i in range(num_agents):
		test_case[i,5] = (radius_bnds[1] - radius_bnds[0]) * np.random.rand() + radius_bnds[0]
		test_case[i,4] = (speed_bnds[1] - speed_bnds[0]) * np.random.rand() + speed_bnds[0]

		while True:
			y 		= Config.MAP_XW/2 - 2 + (np.random.rand() * 2 - 1)
			x_start = Config.MAP_XW/2 - 2 + (np.random.rand() * 2 - 1)
			x_end   = Config.MAP_XW/2 - 2 + (np.random.rand() * 2 - 1)
			sign = 1 if np.random.rand() < 0.5 else -1
			start = np.array([x_start, sign*y]) 
			end   = np.array([x_end,  -sign*y]) 
			
			if static_map.check_feasible(start) or static_map.check_feasible(end):
				continue

			if_collide = False
			for j in range(i):
				radius_start = test_case[j,5] + test_case[i,5] + Config.GETTING_CLOSE_RANGE
				radius_end   = test_case[j,5] + test_case[i,5] + Config.GETTING_CLOSE_RANGE
				if np.linalg.norm(start - test_case[j,0:2] ) < radius_start or np.linalg.norm(end - test_case[j,2:4]) < radius_end:
					if_collide = True
					break

			if if_collide:
				continue

			break

		test_case[i,0:2] = start
		test_case[i,2:4] = end
	return test_case


#######
def cadrl_test_case_to_agents(test_case, policy_to_ensure=None, policies='GA3C_CADRL', policy_distr=None,
                              agents_dynamics='unicycle', agents_sensors=['other_agents_states'], prev_agents=None, static_map=None):
    ###############################
    # policies: either a str denoting a policy everyone should follow
    # This function accepts a test_case in legacy cadrl format and converts it
    # into our new list of Agent objects. The legacy cadrl format is a list of
    # [start_x, start_y, goal_x, goal_y, pref_speed, radius] for each agent.
    ###############################
    if Config.DEBUG: print('test_case ==============='); print(test_case)
    num_agents = np.shape(test_case)[0]
    agents = []
    if type(policies) == str:
        # Everyone follows the same one policy
        agent_policy_list = [policies for _ in range(num_agents)]
    elif type(policies) == list:
        if policy_distr is None:
            # No randomness in agent policies (1st agent gets policies[0], etc.)
            assert(len(policies)==num_agents)
            agent_policy_list = policies
        else:
            # Random mix of agents following various policies
            assert(len(policies)==len(policy_distr))
            agent_policy_list = np.random.choice(policies, num_agents, p=policy_distr)
            if policy_to_ensure is not None and policy_to_ensure not in agent_policy_list:
                # Make sure at least one agent is following the policy_to_ensure
                #  (otherwise waste of time...)
                random_agent_id = np.random.randint(len(agent_policy_list))
                agent_policy_list[random_agent_id] = policy_to_ensure
    else:
        print('Only handle str or list of strs for policies.')
        raise NotImplementedError
  
    agent_dynamics_list = [agents_dynamics for _ in range(num_agents)]
    if not Config.COLLECT_DATA:
        agent_sensors_list = [[sensor_dict[sensor] for sensor in agents_sensors] if agent_policy_list[_] == "learning_ga3c" or 'GA3C_CADRL' else [] for _ in range(num_agents)]
    else:
        agent_sensors_list = [[sensor_dict[sensor] for sensor in agents_sensors] for _ in range(num_agents)]

    for i, agent in enumerate(test_case):
        px = agent[0]
        py = agent[1]
        gx = agent[2]
        gy = agent[3]
        pref_speed = agent[4]
        radius = agent[5]
        if Config.EVALUATE_MODE:
            # initial heading is pointed toward the goal
            vec_to_goal = np.array([gx, gy]) - np.array([px, py])
            heading = np.arctan2(vec_to_goal[1], vec_to_goal[0])
        else:
            heading = np.random.uniform(-np.pi, np.pi)
        policy_str = agent_policy_list[i]
        dynamics_str = agent_dynamics_list[i]
        sensors = agent_sensors_list[i]

        if prev_agents is not None and policy_str == prev_agents[i].policy.str:
            prev_agents[i].reset(px=px, py=py, gx=gx, gy=gy, pref_speed=pref_speed, radius=radius, heading=heading)
            agents.append(prev_agents[i])
        else:
            new_agent = Agent(px, py, gx, gy, radius, pref_speed, heading, policy_dict[policy_str], dynamics_dict[dynamics_str], sensors, i, static_map)
            agents.append(new_agent)
    return agents


# A B C D E D C B A
