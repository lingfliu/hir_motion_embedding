import time

from dataloader import BaseLoader
import os
import h5py
import pickle
import scipy as sp
import numpy as np
from utils import OrderedPool
# from utils import rotate_matrix
import quaternion
import transforms3d

DEFAULT_META_FILE = 'cmu.meta'
DEFAULT_DATA_CACHE_SUFFIX = '.dat'

"""
not used
"""
def amc_frame2bvh_frame(frame, joint_idx, joint_axis, joint_dof):
    frame_pos = []
    for j in frame.keys():
        idx = joint_idx[j]
        axis = joint_axis
        dof = joint_dof[idx]
        rot = [0, 0, 0]
        for i, r in enumerate(frame[j]):
            rot[i] = r

        frame_pos.append([axis, rot])
    return frame_pos

"""
not used
"""
def hierarchy2adj(hierarchy):
    adj = np.zeros((len(hierarchy), len(hierarchy)))
    for i, h in enumerate(hierarchy):
        for j in h:
            adj[i, j] = 1
    return adj


def frame2quat(frame, hierarchy):
    """
    convert amc motion frame (in euler degrees) into quaternion
    :param frame:
    :param hierarchy:
    :return:
    """
    pos_root = np.zeros((3, 1), dtype=np.float32)
    quat_list = np.zeros((len(hierarchy.joints.keys()), 4))
    # offset = hierarchy.joint_offset
    idx_rec = []

    for jname in frame.keys():
        joint = hierarchy.joints[jname]
        jpos = frame[jname]

        # axis is equivalent to rmat in hierarchy posture calculation, not used for amc recording
        axis = joint.axis
        if jname == 'root':
            # root position as [x,y,z, rx, ry, rz]
            pos_root = np.array(jpos[:3])
            deg = jpos[3:]
        else:
            # others as [rx, ry, rz]
            deg = [0, 0, 0]
            cnt = 0
            for i, d in enumerate(joint.dof):
                if d > 0:
                    deg[i] = jpos[cnt]
                    cnt += 1

        deg = np.deg2rad(deg)  # np.array([deg])*math.pi/180

        jidx = hierarchy.joint_idx[jname]
        idx_rec.append(jidx)

        quat = transforms3d.euler.euler2quat(*deg)
        quat_list[jidx, :] = np.array(quat)

    # fill unrecorded joints
    for i in range(hierarchy.joint_idx.keys().__len__()):
        if i not in idx_rec:
            # quaternion
            quat_list[i, :] = np.array([1, 0, 0, 0])

            # offset
            jname = None
            for name in hierarchy.joint_idx.keys():
                if hierarchy.joint_idx[name] == i:
                    jname = name
                    break
            joint = hierarchy.joints[jname]

    return quat_list, pos_root


def frame2pos(frame, hierarchy):
    """
    convert amc motion frame (in euler degrees) into cartesian coordinate
    """
    pos_root = np.zeros((3, 1), dtype=np.float32)
    rmat_list = np.zeros((len(hierarchy.joints.keys()), 3, 3))
    idx_rec = []
    for jname in frame.keys():
        joint = hierarchy.joints[jname]
        jpos = frame[jname]

        # axis is equivalent to rmat in hierarchy posture calculation, not used for amc recoding
        axis = joint.axis
        if jname == 'root':
            #root position as [x,y,z, rx, ry, rz]
            pos_root = jpos[:3]
            deg = jpos[3:]
        else:
            #others as [rx, ry, rz]
            deg = [0, 0, 0]
            cnt = 0
            for i, d in enumerate(joint.dof):
                if d > 0:
                    deg[i] = jpos[cnt]
                    cnt += 1

        deg = np.deg2rad(deg)  # np.array([deg])*math.pi/180

        jidx = hierarchy.joint_idx[jname]
        idx_rec.append(jidx)

        # todo: check rotate_matrix
        # rmat = rotate_matrix(deg[2], deg[0], deg[1])
        rmat = transforms3d.euler.euler2mat(*deg)

        # amat = rotate_matrix(axis[2], axis[0], axis[1])
        amat = joint.rmat
        amat_inv = joint.rmat_inv

        # ref: https://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/ASF-AMC.html
        rmat = amat.dot(rmat).dot(amat_inv)

        rmat_list[jidx, :, :] = rmat

    # fill unrecorded joints
    for jname in hierarchy.joints.keys():
        jidx = hierarchy.joint_idx[jname]
        if not jidx in idx_rec:
            # rotation matrix
            rmat_list[jidx, :, :] = np.eye(3, dtype=np.float32)

    # calculate the positions
    pos = []
    for jname in hierarchy.joints:
        rmat_parent = np.eye(3, dtype=np.float32)
        if jname == 'root':
            pos.append(np.transpose([pos_root]))
        else:
            calc_order = hierarchy.calc_order[jname]
            # root rmat
            rmat_parent = rmat_list[calc_order[-1], :, :]
            # root offset
            jidx = calc_order[-1]
            offset = np.transpose(hierarchy.joint_offset[hierarchy.joint_idx_rev[jidx]])

            # pos in child coordinate = pos_parent + rmat_parent @ rmat @ offset_parent
            p = np.transpose([pos_root]) + rmat_parent @ offset
            for idx in calc_order[::-1]:
                if idx == calc_order[-1]:
                    continue
                rmat = rmat_list[idx, :, :]
                offset = np.transpose(hierarchy.joint_offset[hierarchy.joint_idx_rev[idx]])

                # pos in child coordinate = pos_parent + rmat_parent @ rmat_child @ offset_parent
                p = p + rmat_parent @ rmat @ offset
                # rmat_parent is the cumulative rotation matrix walking through the hierarchy
                rmat_parent = rmat_parent @ rmat
            pos.append(p)

    pos = np.array(pos)
    return pos


def parse_asf(file_path):
    with open(file_path) as f:
        content = f.read().splitlines()

        in_bonedata = False
        in_hierarchy = False
        in_root = False

        root_lines = []
        joint_lines = []
        hierarchy_lines = []
        is_begin = False
        joints = {}
        hierarchy = {}
        joint_idx = {}
        joint_idx_rev = {}

        def parse_root(lines):
            name = 'root'
            axis = [float(x) for x in lines[2].split()[1:]]
            direction = [float(x) for x in lines[3].split()[1:]]

            root = CmuJoint(name, axis, direction, 0)
            return root

        for idx, line in enumerate(content):
            if line.strip() == ':root':
                in_root = True
                in_bonedata = False
                in_hierarchy = False

                continue

            if line.strip() == ':bonedata':
                in_root = False
                in_bonedata = True
                in_hierarchy = False
                is_begin = False
                if 'root' not in joints.keys():
                    joint_root = parse_root(root_lines)
                    joints['root'] = joint_root


            elif line.strip() == ':hierarchy':
                in_root = False
                in_bonedata = False
                in_hierarchy = True
                is_begin = False

                if 'root' not in joints.keys():
                    joint_root = parse_root(root_lines)
                    joints['root'] = joint_root

                # generate joint_idx & joint_idv_rev
                for i, j in enumerate(joints.keys()):
                    joint_idx[j] = i
                    joint_idx_rev[i] = j

                continue
            else:
                # searching the begin marker
                if line.strip() == 'begin':
                    is_begin = True
                    if in_bonedata:
                        joint_lines = []
                    elif in_hierarchy:
                        hierarchy_lines = []
                    continue

                # when end marker is found, parse the data
                elif line.strip() == 'end':
                    is_begin = False
                    if in_bonedata:
                        # convert joint_lines to joint
                        # id = joint_lines[0].split()[1] # not used
                        name = joint_lines[1].split()[1]
                        direction = [float(x) for x in joint_lines[2].split()[1:]]
                        length = float(joint_lines[3].split()[1])
                        axis = [float(x) for x in joint_lines[4].split()[1:4]]

                        dof = [0, 0, 0]
                        limits = [[0, 0, 0]] * 3
                        if len(joint_lines) > 5:  # dof defined
                            dof_names = [x for x in joint_lines[5].split()[1:]]
                            if 'rx' in dof_names:
                                dof[0] = 1
                            if 'ry' in dof_names:
                                dof[1] = 1
                            if 'rz' in dof_names:
                                dof[2] = 1

                            cnt = 0
                            for i, d in enumerate(dof):
                                if d == 1:
                                    limits[i] = (joint_lines[6 + cnt].split('(')[1].split(')')[0].split(','))
                                    cnt += 1

                        joints[name] = CmuJoint(name, axis, direction, length, dof, limits)

                    elif in_hierarchy:
                        hierarchy = {}
                        # construct invert hierarchy list as: {child: [parent]}
                        # construct adjascent matrix
                        hierarchy['root'] = 'root'  # root is the root of the tree
                        hierarchy_mat = np.zeros((len(joint_idx), len(joint_idx)))
                        for line in hierarchy_lines:
                            names = line.split()
                            for name in names[1:]:
                                hierarchy[name] = names[0]

                            root_idx = joint_idx[names[0]]
                            for name in names:
                                hierarchy_mat[root_idx, joint_idx[name]] = 1
                else:

                    if is_begin and in_bonedata:
                        joint_lines.append(line)
                    elif is_begin and in_hierarchy:
                        hierarchy_lines.append(line)
                    elif in_root:
                        root_lines.append(line)

        print("parsed hierarchy", file_path)
        hierarchy = CmuHierarchy(file_path.split('.')[0], joints, joint_idx, joint_idx_rev, hierarchy,
                                 sp.sparse.coo_matrix(hierarchy_mat))
        return hierarchy


def parse_amc(amc_name, amc_path, hierarchy):
    with open(amc_path) as f:
        content = f.read().splitlines()
        is_data = False
        frames = []
        idx_frame = 0
        degrees = {}
        first_frame = True

        for idx, line in enumerate(content):
            if line == ':DEGREES':
                is_data = True
                continue
            else:
                if is_data:
                    data = line.split()
                    if data[0].isnumeric():
                        if first_frame:
                            # skip previous frame save for first frame
                            first_frame = False
                        else:
                            #else save previous frame
                            frames.append((idx_frame, degrees))
                        # reset frame buff
                        idx_frame = int(data[0]) - 1
                        degrees = {}
                    else:
                        degrees[data[0]] = [float(deg) for deg in data[1:]]

        pos_frames = []
        # idx = 0
        for frame in frames:
            pos = frame2pos(frame[1], hierarchy)
            pos_frames.append(pos)

            # print('parsed frame' + str(idx))
            # idx +=1

        print('parsed', amc_name)
        return (amc_name, frames, pos_frames)


def load_amc_cache(cache_file_path):
    with open(cache_file_path, 'rb') as f:
        print("loaded", cache_file_path)
        return pickle.load(f)


def load_asf_cache(cache_file_path):
    with open(cache_file_path, 'rb') as f:
        print("loaded", cache_file_path)
        return pickle.load(f)


class CmuLoader(BaseLoader):
    """
    CMU mocap dataset loader
    cache directory structure:
        cache_root_path / meta / cmu_mocap.meta the meta cache
        cache_root_path / *.dat the mocap data

    :param root_path: the root directory of the CMU dataset
    :param cache_root_path: the directory to store the cache files
    """
    def __init__(self, root_path=None, cache_root_path=None):
        super(CmuLoader, self).__init__()
        # self.task_pool = OrderedPool(10000)
        self.root_path = root_path
        self.cache_root_path = cache_root_path

    def get_meta_path(self):
        return os.path.join(self.cache_root_path, 'meta', DEFAULT_META_FILE)

    def save_meta(self, meta):
        if self.cache_root_path:
            if not os.path.exists(os.path.join(self.cache_root_path, 'meta')):
                os.makedirs(os.path.join(self.cache_root_path, 'meta'))
            with open(os.path.join(self.cache_root_path, 'meta', DEFAULT_META_FILE), 'wb') as f:
                pickle.dump(meta, f)
                return 0
        else:
            return -1

    def load_meta(self):
        """
        :param cache_path:
        :return: meta info
        """
        meta_path = os.path.join(self.cache_root_path, 'meta', DEFAULT_META_FILE)
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            return meta

        mapping = {}
        hierarchies = {}

        for (root, dirs, files) in os.walk(self.root_path, topdown=True):
            asf_name = None
            asf_path = None
            amc_names = []
            amc_paths = []
            for f in files:
                fname = f.split('.')[0]
                fsuffix = f.split('.')[-1]
                if fsuffix == 'amc':
                    amc_names.append(fname)
                    amc_path = os.path.join(root, f)
                    amc_paths.append(amc_path)
                elif fsuffix == 'asf':
                    asf_name = fname
                    asf_path = os.path.join(root, f)
            if asf_name and amc_names:
                for i, amc_name in enumerate(amc_names):
                    mapping[amc_name] = [amc_name, amc_paths[i], asf_name, asf_path]
                hierarchy = parse_asf(asf_path)
                hierarchies[asf_name] = hierarchy

        meta = CmuMeta(self.root_path, mapping, hierarchies)

        return meta

    def load_data_all(self, meta=None):
        if not meta:
            return {}

        mapping = meta.mapping
        hierarchies = meta.hierarchies

        tic = time.time()
        order = 0
        task_pool = OrderedPool(10000)

        for amc_name in mapping.keys():
            _, amc_path, asf_name, asf_path = mapping[amc_name]
            hierarchy = hierarchies[asf_name]
            task_pool.submit(task=parse_amc, order=order, params=(amc_name, amc_path, hierarchy,))
            order += 1

            #todo: test code, remove on release
            if order > 50:
                break

        task_pool.subscribe()
        parsed_list = task_pool.fetch_results()
        task_pool.cleanup()
        toc = time.time()

        parsed_map = {}
        for parsed in parsed_list:
            (amc_name, frames, pos_frames) = parsed
            parsed_map[amc_name] = (frames, pos_frames)

        print("fetching results takes", toc - tic, "seconds")
        print("total data size:", len(parsed))

        return parsed_map

    def load_data(self, amc_name, amc_path, hierarchy):
        if self.cache_root_path:
            data_cache_path = os.path.join(self.cache_root_path, amc_name + DEFAULT_DATA_CACHE_SUFFIX)
            if os.path.exists(data_cache_path):
                frames, pos_frames = load_amc_cache(data_cache_path)
            else:
                _, frames, pos_frames = parse_amc(amc_name, amc_path, hierarchy)
        else:
            _, frames, pos_frames = parse_amc(amc_name, amc_path, hierarchy)

        return frames, pos_frames

    def save_data(self, amc_name, frames, pos_frames):
        if self.cache_root_path:
            cache_path = os.path.join(self.cache_root_path, 'cache')
            if not os.path.exists(cache_path):
                os.makedirs(cache_path)
            data_cache_path = os.path.join(self.cache_root_path, 'cache', amc_name + DEFAULT_DATA_CACHE_SUFFIX)
            with open(data_cache_path, 'wb') as f:
                pickle.dump((frames, pos_frames), f)
                return data_cache_path
        else:
            return None
    
    def frames2quat(self, frames, hierarchy):
        """
        :param frames: [idx, {joint_name: [deg]}]
        :param hierarchy:
        :return: quats: [[w, x, y, z]]
        """
        quats = []
        root_poses = []
        for frame in frames:
            quat, root_pos = frame2quat(frame[1], hierarchy)
            quats.append(quat)
            root_poses.append(root_pos)
        return np.array(quats), np.array(root_poses)

    def frames2pos(self, frames, hierarchy):
        pos = []
        offsets = self.calc_offsets(hierarchy)
        for frame in frames:
            pos.append(frame2pos(frame[1], offsets, hierarchy))
        return pos

class CmuJoint:
    def __init__(self, name, axis, direction, length, dof=None, limits=None):
        if limits is None:
            limits = [[0, 0], [0, 0], [0, 0]]
        if dof is None: # [x, y, z]
            dof = [0, 0, 0]

        self.axis = axis
        self.name = name
        self.limits = limits
        self.dof = dof
        self.direction = direction
        self.length = length

        self.rmat = np.zeros((3, 3))
        self.rmat_inv = np.zeros((3, 3))
        self.calc_rmat()

    def calc_rmat(self):
        deg = np.deg2rad(self.axis)
        self.rmat = transforms3d.euler.euler2mat(*deg)

        # todo: to be tested
        # self.rmat = rotate_matrix(deg[2], deg[0], deg[1])

        self.rmat_inv = np.linalg.inv(self.rmat)


# todo 生成cmu的语义标签
class CmuLabel:
    def __init__(self, labels):
        self.labels = labels

    # demo of segment label
    def segment(self):
        return [[[0, 100], ['run', 'squat']], [[100, 200], ['walk']]]

    def category(self):
        return ['run', 'walk', 'squat']

    def part_segment(self, part):
        return [[[0, 100], ['LA', 'RA'], ['write']]]



class CmuMeta:
    """
    meta data of cmu dataset, including:
    1. amc-asf file_paths mapping
    2. asf hierarchy, joint limit (not used)
    3. summaries
    """

    def __init__(self, root_path, mapping={}, hierarchies={}):
        self.root_path = root_path
        self.mapping = mapping # {amc_name: [amc_name, amc_path, asf_name, asf_path]}
        self.hierarchies = hierarchies # {asf_names: hierarchy}

    def get_hierarchy(self, amc_name):
        asf_name = self.mapping[amc_name][0]
        return self.hierarchy[asf_name]


class CmuHierarchy:
    """
    hierarchy represented in tree structure
    """

    def __init__(self, asf_name, joints, joint_idx, joint_idx_rev, hierarchy, hierarchy_mat):
        self.name = asf_name
        self.joints = joints # CmuJoint
        self.joint_idx = joint_idx
        self.joint_idx_rev = joint_idx_rev
        self.joint_offset = {}

        # invert hierarchy list as: {child: parent}
        self.hierarchy = hierarchy
        self.hierarchy_mat = hierarchy_mat

        self.calc_order = {}

        self.gen_calc_order()

        self.gen_joint_offset()

    def gen_joint_offset(self):
        for jname in self.hierarchy.keys():
            joint = self.joints[jname]
            direction = np.array([joint.direction])
            offset = joint.length * direction
            self.joint_offset[jname] = offset
    """
    预先生成旋转矩阵反向计算序列(从末关节反向追溯，直到root)
    """
    def gen_calc_order(self):
        for jname in self.hierarchy.keys():
            order = [self.joint_idx[jname]]
            if jname == 'root':
                self.calc_order[jname] = order
                continue

            parent = self.hierarchy[jname]
            order.append(self.joint_idx[parent])
            while parent != 'root':
                parent = self.hierarchy[parent]
                order.append(self.joint_idx[parent])

            self.calc_order[jname] = order

    def get_parent(self, child):
        return self.hierarchy[child]
