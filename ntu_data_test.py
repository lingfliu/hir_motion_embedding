import os
from dataloader import NtuLoader, NTU_HIERARCHY
from dataviewer import ntu_draw_pos_motion
import matplotlib.pyplot as plt
import time


if __name__ == '__main__':
    root_path = os.path.join('..','..','data','ntu')
    cache_root_path = os.path.join('..','..','data','ntu','cache')

    loader = NtuLoader(root_path, cache_root_path)

    meta = loader.load_meta()

    '''test code for all motion file'''
    result = loader.cache_data_all(meta)
    # for amc_name in parsed_map.keys():
    #     loader.save_data(amc_name, parsed_map[amc_name][0], parsed_map[amc_name][1])

    '''test code for single motion file parsing'''
    # for ntu_name in meta['mapping'].keys():
    #     source_path = meta['mapping'][ntu_name][1]
    #     frams_map, pos_frames_map = loader.load_data(ntu_name, source_path)
    #
    #     for cmu_name in pos_frames_map.keys():
    #         ntu_draw_pos_motion(pos_frames_map[cmu_name], NTU_HIERARCHY)
    #         break
        # break





    #     amc_path = meta.mapping[amc_name][1]
    #     asf_name = meta.mapping[amc_name][2]
    #     hierarchy = meta.hierarchies[asf_name]
    #
    #     tic = time.time()
    #     frames, pos_frames = loader.load_data(amc_name, amc_path, hierarchy)
    #     toc = time.time()
    #     print('load pos time: ', toc-tic)
    #
    #     tic = time.time()
    #     quat_frames = loader.frames2quat(frames, hierarchy)
    #     toc = time.time()
    #     print('load quat time: ', toc-tic)
    #
    #
    #     # draw_hierarchy(hierarchy)
    #     draw_pos_motion(pos_frames, hierarchy)
    #     # draw_motion(frames, hierarchy)
    #     # loader.save_data(amc_name, frames, pos_frames)
    #     # break

    '''test code for saved data validating'''
    # for amc_name in meta.mapping.keys():
    #     if loader.cache_root_path:
    #         _, amc_path, asf_name, asf_path = meta.mapping[amc_name]
    #         frames, pos_frames = loader.load_data(amc_name, amc_path, meta.hierarchies[asf_name])
    #         draw_pos_motion(pos_frames, meta.hierarchies[meta.mapping[amc_name][2]])
