import re
import numpy as np
from common.quaternion import *
from visualization.Animation import Animation

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}

def load(filename, start=None, end=None, world=False, need_quater=True):
    """
    Reads a BVH file and constructs an animation
    Parameters
    ----------
    filename: str
        File to be opened
    start : int
        Optional Starting Frame
    end : int
        Optional Ending Frame
    order : str
        Optional Specifier for joint order.
        Given as string E.G 'xyz', 'zxy'
    world : bool
        If set to true euler angles are applied
        together in world space rather than local
        space
    Returns
    -------
    (animation, joint_names, frametime)
        Tuple of loaded animation and joint names
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    names = []
    orients = Quaterions.id(0)
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)
    orders = []

    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        # """ Modified line read to handle mixamo data """
        rmatch = re.match(r"ROOT (\w+)", line)
        # rmatch = re.match(r"ROOT (\w+:?\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:
            channels = int(chanmatch.group(1))

            channelis = 0 if channels == 3 else 3
            channelie = 3 if channels == 3 else 6
            parts = line.split()[2 + channelis:2 + channelie]
            if any([p not in channelmap for p in parts]):
                continue
            order = "".join([channelmap[p] for p in parts])
            orders.append(order)
            continue

        # """ Modified line read to handle mixamo data """
        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        # jmatch = re.match("\s*JOINT\s+(\w+:?\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))
            jnum = len(parents)
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(orients), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        # dmatch = line.strip().split(' ')
        dmatch = line.strip().split()
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1

    f.close()

    all_rotations = []
    canonical_order = 'xyz'
    for i, order in enumerate(orders):
        rot = rotations[:, i:i + 1]
        if need_quater:
            quat = euler_to_quat_np(np.radians(rot), order=order, world=world)
            all_rotations.append(quat)
            continue
        elif order != canonical_order:
            quat = euler_to_quat_np(np.radians(rot), order=order, world=world)
            rot = np.degrees(qeuler_np(quat, order=canonical_order))
        all_rotations.append(rot)
    rotations = np.concatenate(all_rotations, axis=1)

    return Animation(rotations, positions, orients, offsets, parents, names, frametime)

def write_bvh(parent, offset, rotation, rot_position, names, frametime, order, path, endsite=None):
    file = open(path, 'w')
    frame = rotation.shape[0]
    assert rotation.shape[-1] == 3
    joint_num = rotation.shape[1]
    order = order.upper()

    file_string = 'HIERARCHY\n'

    seq = []

    def write_static(idx, prefix):
        nonlocal parent, offset, rotation, names, order, endsite, file_string, seq
        seq.append(idx)
        if idx == 0:
            name_label = 'ROOT ' + names[idx]
            channel_label = 'CHANNELS 6 Xposition Yposition Zposition {}rotation {}rotation {}rotation'.format(
                *order)
        else:
            name_label = 'JOINT ' + names[idx]
            channel_label = 'CHANNELS 3 {}rotation {}rotation {}rotation'.format(*order)
        offset_label = 'OFFSET %.6f %.6f %.6f' % (offset[idx][0], offset[idx][1], offset[idx][2])

        file_string += prefix + name_label + '\n'
        file_string += prefix + '{\n'
        file_string += prefix + '\t' + offset_label + '\n'
        file_string += prefix + '\t' + channel_label + '\n'

        has_child = False
        for y in range(idx + 1, rotation.shape[1]):
            if parent[y] == idx:
                has_child = True
                write_static(y, prefix + '\t')
        if not has_child:
            file_string += prefix + '\t' + 'End Site\n'
            file_string += prefix + '\t' + '{\n'
            file_string += prefix + '\t\t' + 'OFFSET 0 0 0\n'
            file_string += prefix + '\t' + '}\n'

        file_string += prefix + '}\n'

    write_static(0, '')

    file_string += 'MOTION\n' + 'Frames: {}\n'.format(frame) + 'Frame Time: %.8f\n' % frametime
    for i in range(frame):
        file_string += '%.6f %.6f %.6f ' % (rot_position[i][0], rot_position[i][1],
                                            rot_position[i][2])
        for j in range(joint_num):
            idx = seq[j]
            file_string += '%.6f %.6f %.6f ' % (rotation[i][idx][0], rotation[i][idx][1], rotation[i][idx][2])
        file_string += '\n'

    file.write(file_string)
    return file_string

class WriterWrapper:
    def __init__(self, parents, frametime, offset=None, names=None):
        self.parents = parents
        self.offset = offset
        self.frametime = frametime
        self.names = names

    def write(self, filename, rot, r_pos, order, offset=None, names=None, repr='quat'):
        """
        Write animation to bvh file
        :param filename:
        :param rot: Quaternion as (w, x, y, z)
        :param pos:
        :param offset:
        :return:
        """
        if repr not in ['euler', 'quat', 'quaternion', 'cont6d']:
            raise Exception('Unknown rotation representation')
        if offset is None:
            offset = self.offset
        if not isinstance(offset, torch.Tensor):
            offset = torch.tensor(offset)
        n_bone = offset.shape[0]

        if repr == 'cont6d':
            rot = rot.reshape(rot.shape[0], -1, 6)
            rot = cont6d_to_quat_np(rot)
        if repr == 'cont6d' or repr == 'quat' or repr == 'quaternion':
#             rot = rot.reshape(rot.shape[0], -1, 4)
#             rot /= rot.norm(dim=-1, keepdim=True) ** 0.5
            euler = qeuler_np(rot, order=order)
            rot = euler

        if names is None:
            if self.names is None:
                names = ['%02d' % i for i in range(n_bone)]
            else:
                names = self.names
        write_bvh(self.parents, offset, rot, r_pos, names, self.frametime, order, filename)