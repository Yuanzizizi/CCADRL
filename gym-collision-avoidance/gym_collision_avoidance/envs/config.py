import numpy as np


class Config(object):
    def __init__(self):
        self.TOPIC                      = "gnn_v0"  
        self.TEST_NOT_LEARNING          = False  
        self.LASER_ON                   = True
        self.RAYS_ON                    = True

        self.NUM_ACTIONS                = 19
        self.NUM_RAYS                   = 18
        self.GS_goal_known              = True

        SELF_STATE    = ['is_learning', 'num_other_agents', 'fmm_dist_to_goal', 'fmm_dist_to_subgoal', 'heading_ego_frame', 'pref_speed', 'radius'] 
        SNESORD_STATE = ['other_agents_states', 'laserscan']
        OTHERS_STATE  = ['other_agents_states', 'obstacle_rays', 'laserscan']
        self.STATES_NOT_USED_IN_AGENT_POLICY = ['is_learning', 'obstacle_rays', 'laserscan']            

        self.STATES_IN_OBS = SELF_STATE+OTHERS_STATE
        self.AGENT_SORTING_METHOD = "closest_first" 


        self.TEST_CASE_FN = "get_testcase_map_random" 
        self.TEST_CASE_ARGS = {
            'policy_to_ensure': 'learning_ga3c',
            'policies':     ['noncoopv1', 'learning_ga3c', 'static'],  
            'policy_distr': [ 0,           1,               0],  
            'speed_bnds':   [1.0, 1.0], 
            'radius_bnds':  [0.3, 0.3], 
            'agents_sensors': SNESORD_STATE} 

        self.DEVICE                     = '/gpu:0' # Device
        self.DISCOUNT                   = 0.97
        # --------------------------------------------------

        self.USE_STATIC_MAP             = True  
        self.UPDATE_FMM_DIST            = False

        ### LOCALMAP OBSERVATIONS PARAM
        self.AGENT_IN_MAP   = False
        self.GRID_CELL_SIZE = 0.04  
        # self.APPROX_RATIO   = 8 
        self.MAP_XW         = 8
        self.MAP_YW         = 8

        ### GAUSS_NOISE
        self.GAUSS_ON   =   False
        self.GAUSS_MEAN =   0
        self.GAUSS_STD  =   0.15
        
        self.USE_PLAN2  =   False

        ### OBSERVATIONAS
        self.COLLECT_DATA = False

        self.FIRST_STATE_INDEX = 1
        self.HOST_AGENT_OBSERVATION_LENGTH = 5 # dist to goal, to subgoal, heading to goal, pref speed, radius
        self.OTHER_AGENT_OBSERVATION_LENGTH = 7 # other px, other py, other vx, other vy, other radius, combined radius, distance between
        if self.GS_goal_known:
            self.OTHER_AGENT_OBSERVATION_LENGTH += 3 # other px, other py, other vx, other vy, other subgx, other subgy, other radius, combined radius, distance between, dist to goal
            
        self.OTHER_AGENT_FULL_OBSERVATION_LENGTH = self.OTHER_AGENT_OBSERVATION_LENGTH
        self.HOST_AGENT_STATE_SIZE = self.HOST_AGENT_OBSERVATION_LENGTH   

        ### GENERAL PARAMETERS
        self.DEBUG               = False # Enable debug (prints more information for debugging purpose)
        self.COLLISION_AVOIDANCE = True
        self.continuous, self.discrete = range(2) # Initialize game types as enum
        self.ACTION_SPACE_TYPE   = self.continuous

        ### DISPLAY
        self.ANIMATE_EPISODES   = False
        self.SHOW_EPISODE_PLOTS = False
        self.SAVE_EPISODE_PLOTS = False
        if not hasattr(self, "PLOT_CIRCLES_ALONG_TRAJ"):
            self.PLOT_CIRCLES_ALONG_TRAJ = True
        self.ANIMATION_PERIOD_STEPS = 1 # plot every n-th DT step (if animate mode on)
        self.PLT_LIMITS = None
        self.PLT_FIG_SIZE = (10, 8)

        ### TRAIN / PLAY / EVALUATE
        self.TRAIN_MODE     = True 
        self.PLAY_MODE      = False 
        self.EVALUATE_MODE  = False 
        self.TRAIN_SINGLE_AGENT = False

        ### SIMULATION
        self.DT                  = 0.2 # seconds between simulation time steps
        self.SUBGOAL_RATIO       = 1
        self.NEAR_GOAL_THRESHOLD = 0.2
        self.MAX_TIME_RATIO      = 3.  # agent has this number times the straight-line-time to reach its goal before "timing out"

        ### REWARDS
        self.REWARD_AT_GOAL                 = 1.0   # reward given when agent reaches goal position
        self.REWARD_COLLISION_WITH_AGENT    = -0.25 # reward given when agent collides with another agent
        self.REWARD_COLLISION_WITH_WALL     = -0.25 # reward given when agent collides with wall
        ## ___ALL___HERE_
        self.REWARD_TOWARDS_GOAL            = 0  # reward given when agent moves towards goal
        self.REWARD_GETTING_CLOSE           = 0.1   # reward when agent gets close to another agent (unused?)
        self.REWARD_ENTERED_NORM_ZONE       = -0.05 # reward when agent enters another agent's social zone
        self.REWARD_TIME_STEP               = 0 # default reward given if none of the others apply (encourage speed)
        self.REWARD_WIGGLY_BEHAVIOR         = 0.0
        self.WIGGLY_BEHAVIOR_THRESHOLD      = np.inf
        self.COLLISION_DIST                 = 0.0   # meters between agents' boundaries for collision
        self.GETTING_CLOSE_RANGE            = 0.25   # meters between agents' boundaries for collision
        # self.SOCIAL_NORMS = "right"
        # self.SOCIAL_NORMS = "left"
        self.SOCIAL_NORMS = "none"

        self.MAX_NUM_OTHER_AGENTS_IN_ENVIRONMENT = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1
        self.MAX_NUM_OTHER_AGENTS_OBSERVED = self.MAX_NUM_AGENTS_IN_ENVIRONMENT - 1

        ### EXPERIMENTS
        self.PLOT_EVERY_N_EPISODES = 100 # for tensorboard visualization

        ### LASER SENSORS
        self.SENSING_HORIZON  = np.inf  ### ___ALL___HERE_
        self.LASERSCAN_LENGTH = 256 # num range readings in one scan
        self.LASERSCAN_NUM_PAST = 1 # num range readings in one scan
        self.laserscan_size = self.LASERSCAN_LENGTH * self.LASERSCAN_NUM_PAST
        self.max_range = 4.0
        self.min_range = 0.0
        self.NUM_STEPS_IN_OBS_HISTORY = 1 # number of time steps to store in observation vector
        self.NUM_PAST_ACTIONS_IN_STATE = 0

        ### RVO AGENTS
        self.RVO_TIME_HORIZON       = 1.0   ## ___ALL___HERE
        self.RVO_TIME_HORIZON_OBST  = 0.0   ## ___ALL___HERE
        self.RVO_COLLAB_COEFF       = 0.5
        self.RVO_ANTI_COLLAB_T      = 1.0

        ### OBSERVATION VECTOR
        self.STATE_INFO_DICT = {
            'is_learning':                  {'dtype': np.float32, 'size': 1, 'bounds': [0., 1.],          'attr': 'get_agent_data_equiv("policy.str", "learning")'}, 
            'num_other_agents':             {'dtype': np.float32, 'size': 1, 'bounds': [0, np.inf],       'attr': 'get_agent_data("num_other_agents_observed")',    'std': np.array([1.],      dtype=np.float32), 'mean': np.array([1.0],    dtype=np.float32)},
            'dist_to_goal':                 {'dtype': np.float32, 'size': 1, 'bounds': [-np.inf, np.inf], 'attr': 'get_agent_data("dist_to_goal")',                 'std': np.array([5.],      dtype=np.float32), 'mean': np.array([0.],     dtype=np.float32)},
            'fmm_dist_to_goal':             {'dtype': np.float32, 'size': 1, 'bounds': [-np.inf, np.inf], 'attr': 'get_agent_data("fmm_dist_to_goal")',             'std': np.array([5.],      dtype=np.float32), 'mean': np.array([0.],     dtype=np.float32)},
            'fmm_dist_to_subgoal':          {'dtype': np.float32, 'size': 1, 'bounds': [-np.inf, np.inf], 'attr': 'get_agent_data("fmm_dist_to_subgoal")',          'std': np.array([self.SUBGOAL_RATIO],      dtype=np.float32), 'mean': np.array([0.],     dtype=np.float32)},
            'heading_ego_frame':            {'dtype': np.float32, 'size': 1, 'bounds': [-np.pi, np.pi],   'attr': 'get_agent_data("heading_ego_frame")',            'std': np.array([3.14],    dtype=np.float32), 'mean': np.array([0.],     dtype=np.float32)},
            'pref_speed':                   {'dtype': np.float32, 'size': 1, 'bounds': [0, np.inf],       'attr': 'get_agent_data("pref_speed")',                   'std': np.array([1.],      dtype=np.float32), 'mean': np.array([1.0],    dtype=np.float32)},
            'radius':                       {'dtype': np.float32, 'size': 1, 'bounds': [0, np.inf],       'attr': 'get_agent_data("radius")',                       'std': np.array([1.],      dtype=np.float32), 'mean': np.array([0.3],    dtype=np.float32)},
            'subgoal_ego_frame':            {'dtype': np.float32, 'size': 2, 'bounds': [-np.inf, np.inf], 'attr': 'get_agent_data("subgoal_ego_frame")',            'std': np.array([0.2, 0.2],dtype=np.float32), 'mean': np.array([0., 0.], dtype=np.float32)},
            'ref_prll_angle_global_frame':  {'dtype': np.float32, 'size': 1, 'bounds': [-np.pi, np.pi],   'attr': 'get_agent_data("ref_prll_angle_global_frame")',  'std': np.array([3.14],    dtype=np.float32), 'mean': np.array([0.],     dtype=np.float32)},
            'other_agents_states': {
                'dtype': np.float32,
                'size': (self.MAX_NUM_OTHER_AGENTS_OBSERVED,self.OTHER_AGENT_FULL_OBSERVATION_LENGTH),
                'bounds': [-np.inf, np.inf],
                'attr': 'get_sensor_data("other_agents_states")',
                'std': np.tile(np.array([5.0, 5.0, 1.0, 1.0, self.SUBGOAL_RATIO, self.SUBGOAL_RATIO, 1.0, 1.0, 5.0, 2.0], dtype=np.float32), (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)) if self.GS_goal_known else
                       np.tile(np.array([5.0, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0], dtype=np.float32), (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),
                'mean': np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0], dtype=np.float32), (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)) if self.GS_goal_known else
                        np.tile(np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.0], dtype=np.float32), (self.MAX_NUM_OTHER_AGENTS_OBSERVED, 1)),            
            },
            'obstacle_rays': {'dtype': np.float32, 'size': self.NUM_RAYS, 'bounds': [-1.5, 1], 'attr': 'get_agent_data("obstacle_rays")', 'std': np.array([1.], dtype=np.float32), 'mean': np.array([0.], dtype=np.float32)},
            'laserscan': {
                'dtype': np.float32,
                'size': (self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH),
                'bounds': [self.min_range, self.max_range],
                'attr': 'get_sensor_data("laserscan")',
                'std':  np.ones((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32),
                'mean': np.zeros((self.LASERSCAN_NUM_PAST, self.LASERSCAN_LENGTH), dtype=np.float32)
            }
        }

        self.local_map_shape =self.local_map_size = self.STATE_INFO_DICT['obstacle_rays']['size']


        self.setup_obs()
        #############################################################################################
        ## DO NOT NEED TO CHANGE USUALLY ############################################################ E

    
    def setup_obs(self):
        self.MEAN_OBS = {}; self.STD_OBS = {}
        for state in self.STATES_IN_OBS:
            if 'mean' in self.STATE_INFO_DICT[state]:
                self.MEAN_OBS[state] = self.STATE_INFO_DICT[state]['mean']
            if 'std' in self.STATE_INFO_DICT[state]:
                self.STD_OBS[state] = self.STATE_INFO_DICT[state]['std']

class Example(Config):
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 10
        self.MAP_FILE = []
        Config.__init__(self)
        # self.UPDATE_FMM_DIST               = True
        self.EVALUATE_MODE = True
        self.TRAIN_MODE = False
        
        # self.SAVE_EPISODE_PLOTS = True
        # self.PLOT_CIRCLES_ALONG_TRAJ = True
        # self.ANIMATE_EPISODES = True

class CollectRegressionDataset(Config): 
    def __init__(self):
        self.MAX_NUM_AGENTS_IN_ENVIRONMENT = 10  #**   
        BASE_DIR = "/home/rl_collision_avoidance/gym-collision-avoidance/gym_collision_avoidance/envs/world_maps/"
        self.MAP_FILE = [BASE_DIR + "map_0.jpg",
                         BASE_DIR + "map_1.jpg", 
                         BASE_DIR + "map_2.jpg", 
                         BASE_DIR + "map_3.jpg",
                         BASE_DIR + "map_4.jpg", 
                         BASE_DIR + "narrow_gate.jpg"]  #** 
        Config.__init__(self)
        self.TEST_CASE_ARGS['policies'] = 'GA3C_CADRL'  #** 
        
        self.TRAIN_NUM_DATAPTS = 100000 if not self.TEST_NOT_LEARNING else 100
        self.TEST_NUM_DATAPTS  = 20000  if not self.TEST_NOT_LEARNING else 20

        self.COLLECT_DATA = True
        self.DATASET_NAME               = self.TOPIC 
        self.AGENT_SORTING_METHOD       = "closest_first"
        self.TRAIN_SINGLE_AGENT         = True
        self.EVALUATE_MODE              = True
        self.TRAIN_MODE                 = False

        # self.SAVE_EPISODE_PLOTS = True
        # self.PLOT_CIRCLES_ALONG_TRAJ = True
        # self.ANIMATE_EPISODES = True


# A B C D E D C B A
