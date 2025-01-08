import numpy as np
from gym_collision_avoidance.envs.policies.InternalPolicy import InternalPolicy


class NonCooperativePolicy_v1(InternalPolicy):
    """ Non Cooperative Agents simply drive at pref speed toward the goal, ignoring other agents. """
    def __init__(self):
        InternalPolicy.__init__(self, str="NonCooperativePolicy")

    def find_next_action(self, obs, agents, i, map=''):
        """ Go at pref_speed, apply a change in heading equal to zero out current ego heading (heading to goal)

        Args:
            obs (dict): ignored
            agents (list): of Agent objects
            i (int): this agent's index in that list

        Returns:
            np array of shape (2,)... [spd, delta_heading]

        """
        seed = np.random.randint(10000)
        np.random.seed(seed)
        action = np.array([agents[i].pref_speed, -agents[i].heading_ego_frame])
        return action

class NonCooperativePolicy_v2(InternalPolicy):
    """ Non Cooperative Agents simply drive at pref speed toward the goal, ignoring other agents. """
    def __init__(self):
        InternalPolicy.__init__(self, str="NonCooperativePolicy")

    def find_next_action(self, obs, agents, i, map=''):
        """ Go at pref_speed, apply a change in heading equal to zero out current ego heading (heading to goal)

        Args:
            obs (dict): ignored
            agents (list): of Agent objects
            i (int): this agent's index in that list

        Returns:
            np array of shape (2,)... [spd, delta_heading]

        """
        seed = np.random.randint(10000)
        np.random.seed(seed)
        action = np.array([agents[i].pref_speed, -agents[i].heading_ego_frame]) if np.random.random() < 0.7 else np.array([0.0, 0.0])
        return action


# A B C D E D C B A
