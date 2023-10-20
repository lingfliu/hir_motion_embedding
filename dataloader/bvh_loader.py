import os
import re

def parse_joint(lines, idx, bracelets=0):
    joint = BvhJoint()
    while idx < len(lines):
        if re.match('^End Site', lines[idx].strip()):
            jname = 'end'
        elif re.match('^JOINT', lines[idx].strip()) or re.match('^ROOT', lines[idx].strip()):
            jname = lines[idx].strip().split(' ')[1].strip()
        else:
            idx += 1
            continue

        joint.name = jname
        idx += 1

        if not re.match('^{', lines[idx].strip()):
            idx += 1
            continue
        # braclet matched
        bracelets += 1
        idx += 1

        name, offset, channels, idx, bracelets = parse_joint_attr(lines, idx, jname, bracelets)
        joint.offset = offset
        joint.channels = channels
        if name: #and bracelets > 0: # children joint matched
            child, idx, bracelets = parse_joint(lines, idx, bracelets)
            joint.children.append(child)
            print('add {0} to {1} child'.format(child.name, joint.name))

    return joint, idx, bracelets




def parse_joint_attr(lines, idx, jname, bracelets):
    """
    parse the first two lines of joint
    """
    joint_name = None
    offset = []
    channels = []
    while idx < len(lines):
        line = lines[idx].strip()
        if re.match('^OFFSET', line):
            offset = [float(x) for x in line.split(' ')[1:]]
            idx += 1
        elif re.match('^CHANNELS', line):
            channels = line.split(' ')[2:]
            idx += 1
        elif re.match('^JOINT', line) :
            joint_name = line.split(' ')[1]
            break
        elif re.match('^End Site', line):
            joint_name = 'end'
            break
        elif re.match('^}', line):
            bracelets -= 1
            break
        else:
            idx += 1

    return joint_name, offset, channels, idx, bracelets

def parse_bvh(source_path):
    """
    parse bvh file
    :param source_path:
    :return: hierarchy, frames, pos_frames
    """
    in_level = -1 # -1: idle, 0: hierarchy, 1: root, 2: joint
    with open(source_path, 'r') as f:
        lines = f.readlines()
        idx = 0
        # parse hierarchy
        while idx < len(lines):
            line = lines[idx].strip()
            if in_level == -1:
                if not re.match('^HIERARCHY', line):
                    idx += 1
                    continue
                else:
                    in_level = 0
                    idx += 1
            # hierarchy matched
            # parse root joint
            line = lines[idx].strip()
            if not re.match('^ROOT ', line):
                idx += 1
                continue
            else:
                # root matched , now parse the hierarchy
                root_joint, idx, bracelets = parse_joint(lines, idx)

                a = 1




class BvhLoader:
    def __init__(self, source_root_path, cache_root_path=None):
        self.source_root_path = source_root_path
        self.cache_root_path = cache_root_path

    def load_data(self, source_file):
        return parse_bvh(os.path.join(self.source_root_path, source_file))


class BvhBody:
    def __init__(self, joints):
        self.joints = joints
        self.hierarchy = self.build_hierarchy()

    def build_hierarchy(self):
        """
        inverse hierarchy
        :return:
        """
        hierarchy = {}
        for joint in self.joints:
            if joint.parent:
                hierarchy[joint] = joint.parent

class BvhJoint:
    def __init__(self, name=None):
        self.name = name
        self.offset = []
        self.direction = []
        self.children = []
        self.parent = None
