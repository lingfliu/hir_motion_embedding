import pickle

from dataloader import BaseLoader

import os
import numpy as np
import h5py

DATA_DIR_60 = '60'
DATA_DIR_120 = '120'


JOINT_NUM = 25 # default joint numbers of ntu skeleton

"""
hierarchy of ntu skeleton, 1 as the root
"""
HIERARCHY = {
    1:1,
    17:1,
    18:17,
    19:18,
    20:19,
    13:1,
    14:13,
    15:14,
    16:15,
    2:1,
    21:2,
    3:21,
    4:3,
    9:21,
    10:9,
    11:10,
    12:11,
    25:12,
    24:12,
    5:21,
    6:5,
    7:6,
    8:7,
    22:8,
    23:8
}


def gen_calc_order(hierarchy):
    """
    预先生成旋转矩阵反向计算序列(从末关节反向追溯，直到root)
    """
    calc_order = {}
    for jidx in hierarchy.keys():
        order = [hierarchy[jidx]]
        if jidx-1 == 0:
            calc_order[jidx] = order
            continue

        p_jidx = hierarchy[jidx]
        order.append(p_jidx)
        while p_jidx-1 != 0:
            p_jidx = hierarchy[p_jidx]
            order.append(p_jidx)

        calc_order[p_jidx] = order
    return calc_order


CALC_ORDER = gen_calc_order(HIERARCHY)


def parse_skeleton(source_path):
    """
    parse skeleton file and return raw frame recording
    :param source_path:
    :return:
    """
    if not os.path.exists(source_path):
        return None

    with(open(source_path, 'r')) as f:
        data = f.readlines()
        num_frames = int(data[0])
        frames = []

        idx = 1
        for idx_frame in range(num_frames):
            # get body count of each frame
            num_bodies = int(data[idx])
            bodies = []

            idx += 1
            for idx_body in range(num_bodies):
                # get joint info for each body
                body_info = data[idx].split(' ')
                bid = str(body_info[0])
                # TODO: leave rest of body_info for future use

                idx += 1
                num_joints = int(data[idx])
                joints = []

                idx += 1
                for k in range(num_joints):
                    joint_info = data[idx].split(' ')
                    x = float(joint_info[0])
                    y = float(joint_info[1])
                    z = float(joint_info[2])
                    depth = [float(joint_info[3]), float(joint_info[4])]
                    color = [float(joint_info[5]), float(joint_info[6])]
                    orientation = [float(joint_info[7]), float(joint_info[8]), float(joint_info[9]), float(joint_info[10])]
                    tracking_state = int(joint_info[11])

                    joint = NtuJoint([x,y,z], depth, color, orientation, tracking_state)
                    joints.append(joint)
                    idx += 1

                body = NtuBody(bid, joints, float(body_info[1]), float(body_info[2]), float(body_info[3]),
                               float(body_info[4]), float(body_info[5]), float(body_info[6]), float(body_info[7]),
                               float(body_info[8]), float(body_info[9]))
                bodies.append(body)
            frames.append(bodies)
    return frames


def filter_pos_frames(pos_frames, invalid_frames):
    """
    filter out abnormal frames
    :param pos_frames:
    :param invalid_frames:
    :return: filtered_pos_frames: np array [num_frames, num_joints, 3]
    """

    for i in range(pos_frames.shape[0]):
        # filter out frame with abnormal aspect ratio
        pos_list = pos_frames[i, :, :]
        x = pos_list[:, 0]
        y = pos_list[:, 1]
        if (x.max() - x.min()) > 0.8 * (y.max() - y.min()) or i in invalid_frames:  # 0.8
            # linear interpolation of invalid frames
            if i == 0:
                pos_frames[i, :, :] = pos_frames[i + 1, :, :]
            elif i == pos_frames.shape[0] - 1:
                pos_frames[i, :, :] = pos_frames[i - 1, :, :]
            else:
                pos_frames[i, :, :] = (pos_frames[i - 1, :, :] + pos_frames[i + 1, :, :]) / 2

    return pos_frames


def calc_dist(pos1, pos2):
    return np.linalg.norm(pos1-pos2)


def pos2angle(pos_frame, hierarchy):
    for jidx in hierarchy.keys():
        parent_jidx = hierarchy[jidx]


def calc_init_root_rotation(pos_frames):
    """
    root rotation defined as vector from spine base to hip center
    :param pos_frames:
    :return:
    """
    orient = (pos_frames[0,16,:] +pos_frames[0,12,:])/2 - pos_frames[0,0,:]

    orient = orient/np.linalg.norm(orient)

    rmat = [[orient[0], orient[1], 0],
               [-orient[1], orient[0], 0],
                [0,0,1]]

    return np.array(rmat)

def normalize_length(pos_frames, hierarchy):
    """
    calculate the bone segment length from the joint to its parent joint
    :param pos_frames:
    :return:
    """

    ref_len = calc_dist(pos_frames[0, :], pos_frames[1, :]) + calc_dist(pos_frames[1,:], pos_frames[20,:])# spine between 0 and 1 + 1 and 20
    joint_lens = [0] * JOINT_NUM

    for jidx in hierarchy.keys():
        parent_jidx = hierarchy[jidx]
        if parent_jidx == 1:
            joint_lens[jidx] = 0
        else:
            joint_lens[jidx] = calc_dist(pos_frames[jidx], pos_frames[parent_jidx])

    joint_lens = np.array(joint_lens)/ref_len

    return joint_lens

def pos2rmat(pos_frame, hierarchy):
    """
    conver pos to rmat & offset
    :param pos_frame:
    :param hierarchy:
    :return:
    """
    rmat_list = np.zeros((JOINT_NUM, 3, 3))
    offset_list = np.zeros((JOINT_NUM, 3))
    for jidx in hierarchy.keys():
        p_idx = hierarchy[jidx]
        pp_idx = hierarchy[p_idx]
        if jidx == 0:
            offset_list[jidx, :] = pos_frame[jidx, :]
            rmat = np.eye((3,3))
            rmat_list[jidx, :, :] = rmat
        else:
            offset = np.array(pos_frame[jidx] - pos_frame[pp_idx])
            offset_list[jidx, :, :] = offset

            offset_prev = np.array(pos_frame[p_idx] - pos_frame[pp_idx])

            # rot is the rotation from offset_prev to offset
            rot = np.cross(offset_prev, offset)
            rot = rot / np.linalg.norm(rot)
            rmat_list[jidx, :, :] = rot

    return rmat_list, offset_list


def rmat2pos(rmat_list, offset_list, hierarchy):
    """
    convert rmat to pos
    :param rmat_list:
    :param offset_list:
    :param hierarchy:
    :return:
    """
    pos_frames = np.zeros((JOINT_NUM, 3))
    for jidx in hierarchy.keys():
        p_jidx = hierarchy[jidx]
        if jidx == 0:
            pos_root = [0, 0, 0]
            pos_frames[jidx, :] = offset_list[0, :]
        else:
            offset = offset_list[jidx, :]
            rot = rmat_list[jidx, :, :]
            pos_frames[jidx, :] = pos_frames[p_jidx, :] + rot@offset

    return pos_frames


def normalize_pos_frames(pos_frames, hierarchy):
    """
    TODO: to be tested
    postuer orientation normalizatio and nody size normalization
    :param pos_frames:
    :return:
    """
    root_rmat = calc_init_root_rotation(pos_frames)

    for i in range(pos_frames.shape[0]):
        # normalized_joint_lens = normalize_length(pos_frames[0, :, :], HIERARCHY)

        # spine between 0 and 1 + 1 and 20
        ref_len = calc_dist(pos_frames[i, 0, :], pos_frames[i, 1, :]) \
                  + calc_dist(pos_frames[i, 1, :], pos_frames[i, 20, :])
        ref_len = ref_len / 2

        for j in range(pos_frames.shape[1]):
            pos_frames[i,j,:] = root_rmat@pos_frames[i,j,:]

            pos_frames[i,j,:] = pos_frames[i,j,:]/ref_len

    return pos_frames


def frame2pos(frames):
    """
    convert frame to pos
    :param frames:
    :return: pos frames: np array [num_frames, num_joints, 3], invalid_frames: list of invalid frame index
    """
    pos_frames = []
    invalid_frames = []
    for i,body in enumerate(frames):
        if body is None:
            invalid_frames.append(i)
            pos_frames.append([[0,0,0]]*25)
        else:

            pos_list = []
            for joint in body.joints:
                pos_list.append(joint.pos)


            pos_frames.append(pos_list)
    pos_frames = np.array(pos_frames)

    return pos_frames, invalid_frames


def frame_format(frames):
    """
    format and validate frames into {bid:frames} where invalid frames are filetered out
    :param frames:
    :return:
    """
    body_id_list = set()
    for frame in frames:
        for body in frame:
            body_id_list.add(body.bid)

    frames_map = {}
    for bid in body_id_list:
        b_frames = []
        for idx, frame in enumerate(frames):
            is_in_frame = False
            for body in frame:
                if body.bid == bid:
                    b_frames.append(body)
                    is_in_frame = True
                    break
            if not is_in_frame:
                b_frames.append(None)

        frames_map[bid] = b_frames

    pos_frames_map = {}
    for bid in frames_map.keys():
        frames = frames_map[bid]
        pos_frames, invalid_frames = frame2pos(frames)
        pos_frames = filter_pos_frames(pos_frames, invalid_frames)

        pos_frames = normalize_pos_frames(pos_frames, HIERARCHY)

        pos_frames_map[bid] = pos_frames

    return frames_map, pos_frames_map


class NtuLoader(BaseLoader):
    """
    NTU RGB+D 60 / 120 skeleton dataset loader
    by default, the 60 and 120 extension data are saved in seperated folders named: 60 & 120
    """
    def __init__(self, root_path, cache_root_path = None):
        self.root_path = root_path
        self.cache_root_path = cache_root_path

    def save_meta(self, meta):
        pass

    def load_meta(self):
        """
        :return: meta: {"source_paths": source_paths, 'mapping': mapping}, where  mapping: {name: (index, path)}
        """
        if self.cache_root_path:
            meta_path = os.path.join(self.cache_root_path, 'ntu.meta')
            if os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    return meta

        source_paths = []
        source_names = []
        for dir in [DATA_DIR_60, DATA_DIR_120]:
            dir_path = os.path.join(self.root_path, dir)
            for file in os.listdir(dir_path):
                if os.path.isfile(os.path.join(dir_path, file)):
                    name, extension = os.path.splitext(file)
                    source_names.append(name)
                    source_paths.append(os.path.join(dir_path, file))

        mapping = {}
        for i, name in enumerate(source_names):
            mapping[name] = (i, source_paths[i])

        meta = {"source_paths": source_paths, 'mapping': mapping}

        files = []
        with open('./matlab/ntu/NTU_RGBD_samples_with_missing_skeletons.txt', 'r') as f:
            [files.append(fname.strip()) for fname in f.readlines()]
        with open('./matlab/ntu/NTU_RGBD120_samples_with_missing_skeletons.txt', 'r') as f:
            [files.append(fname.strip()) for fname in f.readlines()]

        meta['bad_skeletons'] = files

        return meta

    def update_meta(self, meta):
        """
        update: bad data list, dropped frames of each data
        :param meta:
        :return:
        """
        pass

    def load_data_all(self, meta):
        for name in meta['mapping'].keys():
            if self.cache_root_path and os.path.exists(os.path.join(self.cache_root_path,  'data', name+'.data')):
                # 判断缓存文件是否存在
                with open(os.path.join(self.cache_root_path,  'data', 'name.data'), 'rb') as f:
                    pos_frames, frames = pickle.load(f)
            else:
                #  读取ntu数据,生成raw_data缓存文件
                # get_raw_skes_data.get_raw_data_all(self.root_path, meta)

                # 生成denoised_data缓存文件
                # get_raw_denoised_data.get_raw_denoised_data(self.root_path)

                # 返回ntu数据列表
                # pos_frames_list = seq_transformation.seq_transformation(self.cache_dir)

                pass

            # folder_list = meta['ntu_names']
            # return pos_frames_list, folder_list
            return None

    def load_data(self, name, source_path):
        """
        load single ntu data
        :param name: ntu_name
        :param source_path: ntu .skeleton file path
        :return:
        """
        if self.cache_root_path and os.path.exists(os.path.join(self.cache_root_path,  'data', name+'.data')):
            frames_map, pos_frames_map = pickle.load(os.path.join(self.cache_root_path,  'data', name+'.data'))
        else:
            frames = parse_skeleton(source_path)
            frames_map, pos_frames_map = frame_format(frames)

        return frames_map, pos_frames_map

class NtuBody:
    """
    by default a body contains 25 joints,
    see "Amir Shahroudy, Jun Liu, Tian-Tsong Ng, Gang Wang, "NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis", IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016."
    """
    def __init__(self, bid, joints={}, clipedEdges=0, lhConfidence=0, lfState=0, rhConfidence=0, rhState=0, isRestricted=0, lean=0, leanX=0, leanY=0, tracking=0):
        self.bid = bid
        self.joints = joints

        # below not used
        self.clipedEdges = clipedEdges
        self.lhConfidence = lhConfidence
        self.lhState = lfState
        self.rhConfidence = rhConfidence
        self.rhState = rhState
        self.isRestricted = isRestricted
        self.lean = lean
        self.leanX = leanX
        self.leanY = leanY
        self.tracking = tracking

    def export_pos_list(self):
        return np.array([joint.pos for joint in self.joints.values()])



class NtuJoint:
    def __init__(self, pos=[], depth=[], color=[], orientation=[], trackingState=0):
        self.pos = pos
        self.depth = depth
        self.color = color
        self.orientation = orientation
        self.trackingState = trackingState
