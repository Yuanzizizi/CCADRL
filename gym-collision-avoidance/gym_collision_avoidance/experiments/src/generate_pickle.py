import pickle
from PIL import Image
import numpy as np

import math
from random import randint
import torch.nn.functional as F
import torch
from torch import nn


class RoundConv(nn.Module):
    def __init__(self, patch_size=3, in_channels=1 , out_channels=1):
        super(RoundConv, self).__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        x_,y_ = np.mgrid[-patch_size:patch_size+1,-patch_size:patch_size+1]
        kernel = np.exp(-(x_**2+y_**2)/(2 * 5**2))
        threshold = kernel[0, patch_size]
        kernel[kernel >= threshold] = 1
        kernel[kernel < threshold] = 0
        kernel = torch.FloatTensor(kernel).expand(out_channels,1,2*patch_size+1,2*patch_size+1)
        self.weight = nn.Parameter(data=kernel.clone(), requires_grad=False)
        
    def forward(self, x):
        return nn.functional.conv2d(x,self.weight,padding=self.patch_size,groups=self.in_channels)

def conv_array(image_rotation, radius):
    patch_size = int(radius/0.04)
    round_conv = RoundConv(patch_size)
    round_conv_input = torch.FloatTensor(image_rotation.copy()).unsqueeze(0).unsqueeze(0)
    image_rotation = round_conv(round_conv_input).squeeze(0).squeeze(0).numpy()

    # --
    image_rotation[image_rotation<1] = 0
    image_rotation[image_rotation>=1] = 1
    return image_rotation.copy()
    # --
    return np.minimum(image_rotation, 1)

def write_pickle(map_filename, name):
    
    img = Image.open(map_filename) 
    static_map = np.array(img.convert('L'))
    static_map[static_map <  125] = 0
    static_map[static_map >= 125] = 255

    static_map          = np.invert(static_map).astype(bool)
    conv_map            = conv_array(static_map.copy().astype(int), 0.35).astype(bool) 
    phi                 = np.ma.MaskedArray(np.ones(conv_map.shape), conv_map.copy())
    mask_for_init_pos   = conv_array(static_map.copy().astype(int), 0.3).astype(bool)

    print(static_map.shape)
    f = open(name+'.pkl', 'wb')
    content = pickle.dumps([static_map, conv_map, phi, mask_for_init_pos])
    f.write(content)
    f.close()


# A B C D E D C B A
