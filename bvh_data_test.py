import os
from dataloader import BvhLoader
from dataviewer import ntu_draw_pos_motion
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
    root_path = os.path.join('..','..','data','bvh')
    source_file = 'rand_ycx_9Char00.bvh'

    loader = BvhLoader(root_path)

    root_joint = loader.load_data(source_file)

