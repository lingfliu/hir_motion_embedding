import math

import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

from utils import rotate_matrix
# from dataloader import frame2pos

"""draw raw frame by converting to position"""
def draw_frame(axes, frame, hierarchy):
    # pos = frame2pos(frame, hierarchy)
    # draw bone segments
    # draw_pos_frame(axes, pos, hierarchy)

    # plt.show()
    pass


def draw_pos_frame(axes, pos, hierarchy):
    # draw bone segments
    for jname in hierarchy.hierarchy.keys():
        parent_jname = hierarchy.hierarchy[jname]
        idx_j = hierarchy.joint_idx[jname]
        idx_p = hierarchy.joint_idx[parent_jname]
        axes.plot([pos[idx_j][0], pos[idx_p][0]],
                  [pos[idx_j][1], pos[idx_p][1]],
                  [pos[idx_j][2], pos[idx_p][2]],
                  'g', linewidth=1)

    # draw joints
    # axes.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='r', marker='o', s=20)


def config_axes(axes):
    axes.clear()
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    # NotImplementedError: Axes3D目前只支持方面的参数'auto'。你传入了'equal'。
    # axes.set_aspect('equal', adjustable='box')
    axes.set_aspect('auto', adjustable='box')


    axes.set_zlim(-50, 50)
    axes.set_ylim(-10, 40)
    axes.set_xlim(-50, 50)
    axes.view_init(elev=144, azim=-83)


# draw motion by joint positions
def draw_pos_motion(pos_frames, hierarchy):
    plt.ion()
    axes = plt.axes(projection='3d')
    for pos in pos_frames:
        config_axes(axes)
        draw_pos_frame(axes, pos, hierarchy)
        plt.pause(0.001)
        plt.show()


# draw motion from raw data (in degrees), loaded and converted into positions
def draw_motion(data_raw, hierarchy):
    plt.ion()
    axes = plt.axes(projection='3d')

    for frame in data_raw:
        config_axes(axes)
        # draw each frame
        draw_frame(axes, frame[1], hierarchy)
        plt.pause(0.01)


def draw_hierarchy(hierarchy):
    # self.name = name
    # self.joints = joints
    # self.joint_idx = joint_idx
    # self.hierarchy = hierarchy
    # self.hierarchy_mat = hierarchy_mat
    axes = plt.axes(projection='3d')

    pos = []
    rmat_list = []
    offset_list = []
    for i, jname in enumerate(hierarchy.joints.keys()):
        joint = hierarchy.joints[jname]

        offset = np.zeros((3,1), dtype=np.float32)

        # direction+length: the offset of the joint at its local coordinate
        direction = np.array([joint.direction])
        # print(np.linalg.norm(direction))
        # direction = direction / np.linalg.norm(direction)
        length = joint.length

        # axis: the rotation of the joint at its parent coordinate
        axis = np.array(joint.axis)
        axis = axis / math.pi * 180

        rmat = rotate_matrix(axis[2], axis[0], axis[1])
        offset = length*direction

        rmat_list.append(rmat)
        offset_list.append(offset)

    for i, jname in enumerate(hierarchy.joints.keys()):
        if jname == 'root':
            pos.append(np.zeros((3,1)))
            continue

        calc_order = hierarchy.calc_order[jname]
        p = np.transpose([[0,0,0]])
        rmat_parent = rmat_list[calc_order[-1]]
        offset = np.transpose(offset_list[calc_order[-1]])
        p = p + rmat_parent@offset
        for idx in calc_order[::-1]:
            if idx == calc_order[-1]:
                continue
            print(idx)
            rmat = rmat_list[idx]
            offset = np.transpose(offset_list[idx])

            p = p + rmat_parent@offset
            # rmat_parent = rmat@rmat_parent


        pos.append(p)
    pos = np.array(pos)

    # draw bone segments
    draw_pos_frame(axes, pos, hierarchy)

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    axes.set_aspect('equal', adjustable='box')
    plt.show()





