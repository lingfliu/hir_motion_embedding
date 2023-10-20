import os
from dataloader import CmuLoader
from dataviewer import cmu_draw_hierarchy, cmu_draw_pos_motion

if __name__ == '__main__':
    root_path = os.path.join('..','..','data','cmu','all_asfamc','subjects')
    cache_root_path = os.path.join('..','..','data','cmu','cache')


    loader = CmuLoader(root_path, cache_root_path)

    meta = loader.load_meta()

    '''test code for all motion file'''
    # parsed_map = loader.load_data_all(meta)
    # for amc_name in parsed_map.keys():
    #     loader.save_data(amc_name, parsed_map[amc_name][0], parsed_map[amc_name][1])

    '''test code for single motion file parsing'''
    # for amc_name in meta.mapping.keys():
    #     print(amc_name)
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
    for amc_name in meta.mapping.keys():
        if loader.cache_root_path:
            _, amc_path, asf_name, asf_path = meta.mapping[amc_name]
            frames, pos_frames = loader.load_data(amc_name, amc_path, meta.hierarchies[asf_name])
            cmu_draw_pos_motion(pos_frames, meta.hierarchies[meta.mapping[amc_name][2]])
