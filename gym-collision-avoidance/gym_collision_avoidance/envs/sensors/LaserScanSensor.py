import numpy as np
from gym_collision_avoidance.envs.sensors.Sensor import Sensor
from gym_collision_avoidance.envs import Config


class LaserScanSensor(Sensor):
    """ 2D LaserScan based on map of the environment (containing static objects and other agents)

    Currently the laserscan parameters are mostly hard-coded...

    :param num_beams: (int) how many beams/rays should be in the laserscan
    :param num_to_store: (int) how many past laserscans to stack into one measurement
    :param max_range: (float) largest value per beam (meters)
    :param min_range: (float) smallest value per beam (meters)
    :param min_angle: (float) relative to agent's current heading, angle of the first beam (radians)
    :param max_angle: (float) relative to agent's current heading, angle of the last beam (radians)
    """
    def __init__(self):
        if not Config.USE_STATIC_MAP:
            print("LaserScanSensor won't work without static map enabled (Config.USE_STATIC_MAP)")
            assert(0)
        Sensor.__init__(self)
        self.name = 'laserscan'
        self.num_beams = Config.LASERSCAN_LENGTH
        self.num_to_store = Config.LASERSCAN_NUM_PAST
        self.range_resolution = 0.1  # (float) radians between each beam
        self.max_range = Config.max_range
        self.min_range = Config.min_range
        self.min_angle = -np.pi
        self.max_angle =  np.pi

        # (np array) linearly spaced array of angles, ranging from min_angle to max_angle, containing num_beams
        # (np array) linearly spaced array of ranges, ranging from min_range to max_range, spaced by range_resolution
        self.angles = np.linspace(self.min_angle, self.max_angle, self.num_beams)  
        self.ranges = np.arange(self.min_range, self.max_range, self.range_resolution)

        self.measurement_history = np.zeros((self.num_to_store, self.num_beams))
        self.num_measurements_made = 0


    def sense(self, agents, agent_index, top_down_map):
        """ Use top_down_map to ray-trace for obstacles, with sensor located at agents[agent_index] center.

        Args:
            agents (list): all :class:`~gym_collision_avoidance.envs.agent.Agent` in the environment
            agent_index (int): index of this agent (the one with this sensor) in :code:`agents`
            top_down_map (2D np array): binary image with 0 if that pixel is free space, 1 if occupied

        Returns:
            measurement_history (np array): (:code:`num_to_store` x :code:`num_beams`) stacked history of laserscans, where each entry is a range in meters of the nearest obstacle at that angle

        """
        # Approx 200x faster than sense_old (0.002sec per call vs. 0.4sec) :)
        if not Config.LASER_ON:
            return self.measurement_history.copy()
        host_agent = agents[agent_index]

        angles = self.angles + host_agent.heading_global_frame
        ranges = self.ranges
        angles_ranges_mesh = np.meshgrid(angles, ranges)
        angles_ranges = np.dstack([angles_ranges_mesh[0], angles_ranges_mesh[1]])
        beam_coords = np.tile(host_agent.pos_global_frame, (len(angles), len(ranges), 1)).astype(np.float64)
        beam_coords[:,:,0] += (angles_ranges[:,:,1]*np.cos(angles_ranges[:,:,0])).T
        beam_coords[:,:,1] += (angles_ranges[:,:,1]*np.sin(angles_ranges[:,:,0])).T

        iis, jjs, in_maps = top_down_map.world_coordinates_to_map_indices_vec(beam_coords)

        ego_agent_mask = top_down_map.get_agent_mask(host_agent.pos_global_frame, host_agent.radius)
        lidar_hits = np.logical_and.reduce((top_down_map.conv_map.astype(int)[iis, jjs], np.invert(ego_agent_mask[iis, jjs]), in_maps))
        lidar_hits_cumsum = np.cumsum(lidar_hits, axis=1)
        first_hits = np.where(lidar_hits_cumsum == 1)

        ranges = self.max_range*np.ones_like(self.angles)
        ranges[first_hits[0]] = self.ranges[first_hits[1]]

        if self.num_measurements_made == 0:
            self.measurement_history[:,:] = ranges
        else:
            self.measurement_history = np.roll(self.measurement_history, 1, axis=0)
            self.measurement_history[0,:] = ranges

        self.num_measurements_made += 1

        return self.measurement_history.copy()/(Config.max_range-Config.min_range) - 0.5


# A B C D E D C B A
