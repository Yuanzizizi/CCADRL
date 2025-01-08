import numpy as np
import tensorflow as tf
from gym_collision_avoidance.envs import Config

# np.set_printoptions(precision=3, suppress=True)

class Actions():
    def __init__(self, num=Config.NUM_ACTIONS):
        if num == 19:
            # [v_pref,      [-pi/3, -pi/4, -pi/6, -pi/12, 0, pi/12, pi/6, pi/4, pi/3]]
            # [0.5*v_pref,  [-pi/3, -pi/6, 0, pi/6, pi/3]]
            # [0,           [-pi/3, -pi/6, 0, pi/6, pi/3]]
            self.actions = np.mgrid[1.0:1.1:0.5, -np.pi/3:np.pi/3+0.01:np.pi/12].reshape(2, -1).T
            self.actions = np.vstack([self.actions,np.mgrid[0.5:0.6:0.5, -np.pi/3:np.pi/3+0.01:np.pi/6].reshape(2, -1).T])
            self.actions = np.vstack([self.actions,np.mgrid[0.0:0.1:0.5, -np.pi/3:np.pi/3+0.01:np.pi/6].reshape(2, -1).T])
            self.num_actions = len(self.actions)
            assert(self.num_actions == Config.NUM_ACTIONS)
        else:
            print(num, 'Wrong !!!  NUM_ACTIONS')
            assert(0)
            
class NetworkVPCore(object):
    def __init__(self, device, model_name, num_actions):
        assert(Config.RAYS_ON and Config.LASER_ON)
        self.device = device
        self.model_name = model_name
        self.num_actions = num_actions

    def predict_p(self, x):
        return self.sess.run(self.softmax_p, feed_dict={self.x: x[0], self.local_map: x[1], self.laserscan: x[2]})
    def predict_v(self, obs):
        if type(obs) == dict:
            # Turn the dict observation into a flattened vector
            vec_obs = np.array([])
            for state in Config.STATES_IN_OBS:
                if state not in Config.STATES_NOT_USED_IN_AGENT_POLICY:
                    vec_obs = np.hstack([vec_obs, obs[state].flatten()])
            vec_obs = np.expand_dims(vec_obs, axis=0)

        x = [vec_obs, obs['obstacle_rays'].reshape([-1, Config.NUM_RAYS]), obs['laserscan'].reshape([-1, Config.LASERSCAN_NUM_PAST, Config.LASERSCAN_LENGTH])]

        return self.sess.run(self.v, feed_dict={self.x: x[0], self.local_map: x[1], self.laserscan: x[2]})

    def simple_load(self, filename=None):
        if filename is None:
            print("[network.py] Didn't define simple_load filename")
            raise NotImplementedError
        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            with tf.device(self.device):
                self.sess = tf.Session(
                    graph=self.graph,
                    config=tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=tf.GPUOptions(allow_growth=True)))

                new_saver = tf.train.import_meta_graph(filename+'.meta', clear_devices=True)
                self.sess.run(tf.global_variables_initializer())
                new_saver.restore(self.sess, filename)

                self.similarity = g.get_tensor_by_name('Softmax:0')
                self.softmax_p = g.get_tensor_by_name('Softmax_1:0')
               
                self.x = g.get_tensor_by_name('X:0')
                self.local_map = g.get_tensor_by_name('LOCALMAP:0')
                self.laserscan = g.get_tensor_by_name('LASERSCAN:0')

                self.v = g.get_tensor_by_name('Squeeze:0')

class NetworkVP_rnn(NetworkVPCore):
    def __init__(self, device, model_name, num_actions):
        super(self.__class__, self).__init__(device, model_name, num_actions)


# A B C D E D C B A
