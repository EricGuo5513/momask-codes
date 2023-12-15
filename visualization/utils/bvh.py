import re
import numpy as np

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

def load(filename:str, order:str=None) -> dict:
    """Loads a BVH file.
    
    Args:
        filename (str): Path to the BVH file.
        order (str): The order of the rotation channels. (i.e."xyz")
    
    Returns:
        dict: A dictionary containing the following keys:
            * names (list)(jnum): The names of the joints.
            * parents (list)(jnum): The parent indices.
            * offsets (np.ndarray)(jnum, 3): The offsets of the joints.
            * rotations (np.ndarray)(fnum, jnum, 3) : The local coordinates of rotations of the joints. 
            * positions (np.ndarray)(fnum, jnum, 3) : The positions of the joints.
            * order (str): The order of the channels.
            * frametime (float): The time between two frames.
    """

    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False

    # Create empty lists for saving parameters
    names = []
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)

    # Parse the file, line by line
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
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
            if order is None:
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            fnum = int(fmatch.group(1))
            positions = offsets[None].repeat(fnum, axis=0)
            rotations = np.zeros((fnum, len(offsets), 3))
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)
            fi = i
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

    return {
        'rotations': rotations,
        'positions': positions,
        'offsets': offsets,
        'parents': parents,
        'names': names,
        'order': order,
        'frametime': frametime
    }
    
    
def save_joint(f, data, t, i, save_order, order='zyx', save_positions=False):
    
    save_order.append(i)
    
    f.write("%sJOINT %s\n" % (t, data['names'][i]))
    f.write("%s{\n" % t)
    t += '\t'

    f.write("%sOFFSET %f %f %f\n" % (t, data['offsets'][i,0], data['offsets'][i,1], data['offsets'][i,2]))
    
    if save_positions:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    
    end_site = True
    
    for j in range(len(data['parents'])):
        if data['parents'][j] == i:
            t = save_joint(f, data, t, j, save_order, order=order, save_positions=save_positions)
            end_site = False
    
    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)

    t = t[:-1]
    f.write("%s}\n" % t)
    
    return t
    

def save(filename, data, save_positions=False):
    """ Save a joint hierarchy to a file.
    
    Args:
        filename (str): The output will save on the bvh file.
        data (dict): The data to save.(rotations, positions, offsets, parents, names, order, frametime)
        save_positions (bool): Whether to save all of joint positions on MOTION. (False is recommended.)
    """
    
    order = data['order']
    frametime = data['frametime']
    
    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, data['names'][0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % (t, data['offsets'][0,0], data['offsets'][0,1], data['offsets'][0,2]) )
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % 
            (t, channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))

        save_order = [0]
            
        for i in range(len(data['parents'])):
            if data['parents'][i] == 0:
                t = save_joint(f, data, t, i, save_order, order=order, save_positions=save_positions)
    
        t = t[:-1]
        f.write("%s}\n" % t)

        rots, poss = data['rotations'], data['positions']

        f.write("MOTION\n")
        f.write("Frames: %i\n" % len(rots));
        f.write("Frame Time: %f\n" % frametime);
        
        for i in range(rots.shape[0]):
            for j in save_order:
                
                if save_positions or j == 0:
                
                    f.write("%f %f %f %f %f %f " % (
                        poss[i,j,0], poss[i,j,1], poss[i,j,2], 
                        rots[i,j,0], rots[i,j,1], rots[i,j,2]))
                
                else:
                    
                    f.write("%f %f %f " % (
                        rots[i,j,0], rots[i,j,1], rots[i,j,2]))

            f.write("\n")