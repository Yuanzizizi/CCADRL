import numpy as np

from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy
from gym_collision_avoidance.envs.policies.GA3C_CADRL import network
from gym_collision_avoidance.envs import Config
from gym_collision_avoidance.envs import util
from gym_collision_avoidance.envs.Map import Map


class GA3CCADRLPolicy(InternalPolicy):
    def __init__(self):
        InternalPolicy.__init__(self, str="GA3C_CADRL")

        assert(Config.RAYS_ON and Config.LASER_ON)

        self.possible_actions = network.Actions()
        self.nn = network.NetworkVP_rnn(Config.DEVICE, 'network', self.possible_actions.num_actions)
        self.map = Map(x_width=Config.MAP_XW, y_width=Config.MAP_YW, grid_cell_size=Config.GRID_CELL_SIZE, map_filename=Config.MAP_FILE[0])

    def initialize_network(self, **kwargs):  
        """ Load the model parameters of either a default file, or if provided through kwargs, a specific path and/or tensorflow checkpoint.

        Args:
            kwargs['checkpt_name'] (str): name of checkpoint file to load (without file extension)
            kwargs['checkpt_dir'] (str): path to checkpoint

        """
        self.nn.simple_load(kwargs['checkpt_dir'] + kwargs['checkpt_name'])

    # def get_feasible_actions(self, agents, i): 
    #     feasible_actions = []
    #     for aa_idx in range(self.possible_actions.num_actions):
    #         _speed, _heading = self.possible_actions.actions[aa_idx]
    #         _heading = util.wrap(_heading+agents[i].heading_global_frame) 

    #         _collision = False
    #         for t in range(1, 4):
    #             _dx = _speed * np.cos(_heading) * Config.DT * t
    #             _dy = _speed * np.sin(_heading) * Config.DT * t

    #             _pos = agents[i].pos_global_frame + np.array([_dx, _dy])
    #             if self.map.check_collisions(_pos):
    #                 _collision = True
    #                 break
    #         if _collision:
    #             continue
    #         feasible_actions.append(aa_idx)
    #     return feasible_actions

    def find_next_action(self, obs, agents, i, map_filename='', input_data=None, enable=False):
        pref_speed = obs['pref_speed']
        if type(obs) == dict:
            # Turn the dict observation into a flattened vector
            vec_obs = np.array([])
            for state in Config.STATES_IN_OBS:
                if state not in Config.STATES_NOT_USED_IN_AGENT_POLICY:
                    vec_obs = np.hstack([vec_obs, obs[state].flatten()])
            vec_obs = np.expand_dims(vec_obs, axis=0)

        s = [vec_obs, obs['obstacle_rays'].reshape([-1, Config.NUM_RAYS]), obs['laserscan'].reshape([-1, Config.LASERSCAN_NUM_PAST, Config.LASERSCAN_LENGTH])]

        if not Config.RAYS_ON and not Config.LASER_ON:
            predictions = self.nn.predict_p(vec_obs)[0]
        else:
            predictions = self.nn.predict_p(s)[0]

        action_index = np.argmax(predictions) 
        raw_action = self.possible_actions.actions[action_index]

        # feasible_actions = self.get_feasible_actions(agents, i)
        # raw_action = self.possible_actions.actions[feasible_actions[np.argmax(predictions[feasible_actions])]] ## feasible actions --

        action = np.array([pref_speed*raw_action[0], raw_action[1]])
        if not enable:
            return action
        pred = agents[i].sess.run(agents[i].predictions, feed_dict={agents[i].inputs: input_data.reshape(1, 5)})
        return action * np.argmax(pred, axis=1)[0] 


if __name__ == '__main__':
    policy = GA3CCADRLPolicy()
    policy.initialize_network()


# A B C D E D C B A
