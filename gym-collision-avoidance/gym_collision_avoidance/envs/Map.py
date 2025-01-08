import os
import numpy as np
from gym_collision_avoidance.envs.util import *
import pickle


class Map():
    def __init__(self, x_width, y_width, grid_cell_size, map_filename=None):
        self.x_width = x_width
        self.y_width = y_width
        self.grid_cell_size = grid_cell_size
        self.map_filename = map_filename
        self.origin_coords = np.array([(self.x_width/2.)/self.grid_cell_size, (self.y_width/2.)/self.grid_cell_size])

        base_path = os.path.dirname(os.path.abspath(__file__))  
        file_path = os.path.join(base_path, "world_maps/", map_filename.split("/")[-1].split(".")[0] + '.pkl')
        f = open(file_path, 'rb')
        self.static_map, self.conv_map, self.phi, self.mask_for_init_pos = pickle.loads(f.read())
        f.close()

        
    def world_coordinates_to_map_indices(self, pos):
        # for a single [px, py] -> [gx, gy]
        gx = int(np.floor(self.origin_coords[0]-pos[1]/self.grid_cell_size))
        gy = int(np.floor(self.origin_coords[1]+pos[0]/self.grid_cell_size))
        grid_coords = np.array([gx, gy])
        in_map = gx >= 0 and gy >= 0 and gx < self.static_map.shape[0] and gy < self.static_map.shape[1]
        return grid_coords, in_map

    def map_indices_to_world_coordinates(self, indice):
        gx, gy = indice

        pos_y =( self.origin_coords[0]-gx)*self.grid_cell_size 
        pos_x =(-self.origin_coords[1]+gy)*self.grid_cell_size
        
        return pos_x, pos_y

    def world_coordinates_to_map_indices_vec(self, pos):
        # for a 3d array of [[[px, py]]] -> gx=[...], gy=[...]
        gxs = np.floor(self.origin_coords[0]-pos[:,:,1]/self.grid_cell_size).astype(int)
        gys = np.floor(self.origin_coords[1]+pos[:,:,0]/self.grid_cell_size).astype(int)
        in_map = np.logical_and.reduce((gxs >= 0, gys >= 0, gxs < self.static_map.shape[0], gys < self.static_map.shape[1]))
        
        # gxs, gys filled to -1 if outside map to ensure you don't query pts outside map
        not_in_map_inds = np.where(in_map == False)
        gxs[not_in_map_inds] = -1
        gys[not_in_map_inds] = -1
        return gxs, gys, in_map

    def add_agents_to_map(self, agent_pos_global_frame):
        mask = self.get_agent_mask(agent_pos_global_frame, 0.65)
        self.conv_map[mask] = True

        phi = np.ones(self.conv_map.shape)
        self.phi = np.ma.MaskedArray(phi, self.conv_map.astype(bool).copy())


    def get_agent_map_indices(self, pos, radius):
        x = np.arange(0, self.static_map.shape[1])
        y = np.arange(0, self.static_map.shape[0])
        mask = (x[np.newaxis,:]-pos[1])**2 + (y[:,np.newaxis]-pos[0])**2 < (radius/self.grid_cell_size)**2
        return mask

    def get_agent_mask(self, global_pos, radius):
        [gx, gy], in_map = self.world_coordinates_to_map_indices(global_pos)
        if in_map:
            mask = self.get_agent_map_indices([gx,gy], radius)
            return mask
        else:
            assert(0)

    def check_collisions(self, global_pos, dis_lim=0.3):
        pos_index = self.world_coordinates_to_map_indices(global_pos)[0]
        try:
            result = self.mask_for_init_pos[pos_index[0], pos_index[1]]  
        except IndexError:
            return True
        return result

    def check_feasible(self, global_pos, dis_lim=0.3):
        pos_index = self.world_coordinates_to_map_indices(global_pos)[0]
        try:
            result = self.conv_map[pos_index[0], pos_index[1]]  
        except IndexError:
            return True
        return result


# A B C D E D C B A
