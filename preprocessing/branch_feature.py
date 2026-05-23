import numpy as np
import os
import glob
import sys
import pandas as pd

sys.path.append(r'..')
from basicfunc import read_swc, get_soma
import warnings


def n_branch(swcpath, ref_swcpath, typelist):
    branch_info_list = []

    def init_info_dict():
        return {'id': None, 'pid': None,
                'path_distance': None, 'euclidean_distance': None,
                'euc_dist_soma_branch_start': None, 'euc_dist_soma_branch_end': None,
                'branch_level': None,
                'is_terminal': None,
                # 'is_cutoff': None,
                'x_start': None, 'y_start': None, 'z_start': None,
                'x_end': None, 'y_end': None, 'z_end': None,
                'end_local_angle': None, 'end_remote_angle': None
                }

    def gettwoChildlist(idx, indexChildren):
        if len(indexChildren[idx]) > 2:
            # raise ValueError(f'indexChildren excepts 2 items, but receive {len(indexChildren[idx])}')
            warnings.warn(f'indexChildren excepts 2 items, but receive {len(indexChildren[idx])}')
        childlist_all = []
        ch_len = len(indexChildren[idx]) if len(indexChildren[idx]) <= 2 else 2
        for i in range(0, ch_len):
            tmp = indexChildren[idx][i]
            childlist = [tmp]
            while len(indexChildren[tmp]) == 1:
                tmp = indexChildren[tmp][0]
                childlist.append(tmp)
            childlist_all.append(childlist)
        return childlist_all

    def getlocalangle(p0, p1list, p2list, tolerance=0, ensemble_range=5):
        # p1, p2 should consider about the tracing error under the truncated diameter of dendrite branch
        # tolerance means the minimal distance of selecting sampling points
        p0 = np.asarray(p0)
        p1list = np.asarray(p1list)
        p2list = np.asarray(p2list)

        def estimate_point(p0, plist, tolerance=tolerance, ensemble_range=ensemble_range):
            if tolerance >= ensemble_range:
                raise ValueError('ensemble_range should not be greater than tolerance')
            ensemble_list = []
            for i in range(len(plist)):
                tmpp = plist[i]
                dist = np.linalg.norm(p0 - tmpp)
                if 0 < dist < tolerance:
                    if i == len(plist) - 1:
                        return tmpp
                    else:
                        continue
                elif dist <= ensemble_range:
                    ensemble_list.append(tmpp)
                else:
                    break
            if not ensemble_list:
                for i in range(len(plist)):
                    v = plist[i] - p0
                    vlen = np.linalg.norm(v)
                    if vlen == 0: continue
                    v = v / vlen
                    tmpp = p0 + ensemble_range * v
                    break
            else:
                tmpp = np.mean(ensemble_list, axis=0)

            return tmpp

        p1 = estimate_point(p0, p1list)
        p2 = estimate_point(p0, p2list)

        return angle_calc(p0, p1, p2)

    def angle_calc(p0, p1, p2):
        def unit_vector(vector):
            return vector / np.linalg.norm(vector)

        p0 = np.asarray(p0)
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        v1 = p1 - p0
        v2 = p2 - p0
        v1_u = unit_vector(v1)
        v2_u = unit_vector(v2)
        return 180 * np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) / np.pi

    def search_is_cutoff(node, ref_swc):
        '''1 for truncated terminal, 0 for real terminal'''
        pidlist = np.array(ref_swc)[:, 6]
        pos1 = np.round(node[2:5], 2)
        for i in range(len(ref_swc)):
            pos2 = np.round(ref_swc[i][2:5], 2)
            if pos1[0] == pos2[0] and pos1[1] == pos2[1] and pos1[2] == pos2[2]:
                cur_id = ref_swc[i][0]
                if cur_id in pidlist:
                    return 1
                else:
                    return 0
        return 0

    src = read_swc(swcpath, mode=typelist)
    ref_src = read_swc(ref_swcpath, mode=typelist)
    NeuronHash = {}
    orderHash = {}
    branchidHash = {}
    indexChildren = []
    for i in range(len(src)):
        NeuronHash[src[i][0]] = i
        orderHash[src[i][0]] = -1
        branchidHash[src[i][0]] = -1
        indexChildren.append([])
    for i in range(len(src)):
        pid = src[i][6]
        idx = NeuronHash.get(pid)
        if idx is None: continue
        indexChildren[idx].append(i)

    # DBS
    root = get_soma(src)
    root_n = root[0]
    root_idx = NeuronHash[root_n]
    orderHash[root_n] = 0
    cur_branch_id = -1
    branchidHash[root_n] = -1

    bifurs_chs = indexChildren[root_idx]
    while bifurs_chs:
        template_info = init_info_dict()

        cur_idx = bifurs_chs.pop()
        cur_node = src[cur_idx]
        cur_n, cur_pid = cur_node[0], cur_node[6]
        cur_order = orderHash[cur_pid]
        if cur_order >= 10: continue

        template_info['branch_level'] = cur_order + 1
        cur_branch_id += 1
        template_info['id'] = cur_branch_id
        branchidHash[cur_n] = cur_branch_id
        if cur_pid == root_n:
            template_info['pid'] = -1
        else:
            template_info['pid'] = branchidHash[cur_pid]

        cur_branch_length = 0

        cur_child_idx_list = indexChildren[cur_idx]

        pnode = src[NeuronHash[cur_pid]]
        node1_for_euc = pnode

        tmpLength = np.linalg.norm(np.array(cur_node[2:5]) - np.array(pnode[2:5]))
        cur_branch_length += tmpLength

        # if bifurcation, order+1, else order same as parent
        while len(cur_child_idx_list) == 1:
            orderHash[cur_n] = cur_order
            next_idx = cur_child_idx_list[0]
            next_node = src[next_idx]

            tmpLength = np.linalg.norm(np.array(cur_node[2:5]) - np.array(next_node[2:5]))
            cur_branch_length += tmpLength

            cur_idx = next_idx
            cur_node = next_node
            cur_n, cur_pid = cur_node[0], cur_node[6]
            cur_child_idx_list = indexChildren[cur_idx]

            branchidHash[cur_n] = branchidHash[cur_pid]

        if len(cur_child_idx_list) > 1:
            orderHash[cur_n] = cur_order + 1
            bifurs_chs.extend(cur_child_idx_list)
            node2_for_euc = list(cur_node)
            euc = np.linalg.norm(np.array(node2_for_euc[2:5]) - np.array(node1_for_euc[2:5]))
            childidxlist1, childidxlist2 = gettwoChildlist(cur_idx, indexChildren)
            p0 = src[cur_idx][2:5]
            remote_p1 = src[childidxlist1[-1]][2:5]
            remote_p2 = src[childidxlist2[-1]][2:5]
            template_info['end_remote_angle'] = angle_calc(p0, remote_p1, remote_p2)
            template_info['end_local_angle'] = getlocalangle(p0, np.array(src)[np.array(childidxlist1)][:, 2:5],
                                                             np.array(src)[np.array(childidxlist2)][:, 2:5])
            template_info['x_start'] = node1_for_euc[2]
            template_info['y_start'] = node1_for_euc[3]
            template_info['z_start'] = node1_for_euc[4]
            template_info['x_end'] = node2_for_euc[2]
            template_info['y_end'] = node2_for_euc[3]
            template_info['z_end'] = node2_for_euc[4]
            template_info['euc_dist_soma_branch_start'] = np.linalg.norm(
                np.array(src[root_idx][2:5]) - np.array(node1_for_euc[2:5]))
            template_info['euc_dist_soma_branch_end'] = np.linalg.norm(
                np.array(src[root_idx][2:5]) - np.array(node2_for_euc[2:5]))

            v1 = np.array(node1_for_euc[2:5]) - np.array(root[2:5])
            v1_len = np.linalg.norm(v1)
            deviation_angle_soma = None
            if v1_len != 0:
                v1 = v1 / v1_len
                v2 = np.array(node2_for_euc[2:5]) - np.array(node1_for_euc[2:5])
                v2 = v2 / np.linalg.norm(v2)
                deviation_angle_soma = 180 * np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) / np.pi
            template_info['deviation_angle_soma'] = deviation_angle_soma

            template_info['euclidean_distance'] = euc
            template_info['path_distance'] = cur_branch_length
            template_info['is_terminal'] = 0
            template_info['is_cutoff'] = 0

            branch_info_list.append(template_info)

        elif len(cur_child_idx_list) == 0:
            orderHash[cur_n] = cur_order
            node2_for_euc = list(cur_node)
            euc = np.linalg.norm(np.array(node2_for_euc[2:5]) - np.array(node1_for_euc[2:5]))
            template_info['euclidean_distance'] = euc
            template_info['path_distance'] = cur_branch_length
            template_info['is_terminal'] = 1
            template_info['is_cutoff'] = search_is_cutoff(cur_node, ref_src)
            template_info['x_start'] = node1_for_euc[2]
            template_info['y_start'] = node1_for_euc[3]
            template_info['z_start'] = node1_for_euc[4]
            template_info['x_end'] = node2_for_euc[2]
            template_info['y_end'] = node2_for_euc[3]
            template_info['z_end'] = node2_for_euc[4]
            template_info['euc_dist_soma_branch_start'] = np.linalg.norm(
                np.array(src[root_idx][2:5]) - np.array(node1_for_euc[2:5]))
            template_info['euc_dist_soma_branch_end'] = np.linalg.norm(
                np.array(src[root_idx][2:5]) - np.array(node2_for_euc[2:5]))

            v1 = np.array(node1_for_euc[2:5]) - np.array(root[2:5])
            v1_len = np.linalg.norm(v1)
            deviation_angle_soma = None
            if v1_len != 0:
                v1 = v1 / v1_len
                v2 = np.array(node2_for_euc[2:5]) - np.array(node1_for_euc[2:5])
                v2 = v2 / np.linalg.norm(v2)
                deviation_angle_soma = 180 * np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) / np.pi
            template_info['deviation_angle_soma'] = deviation_angle_soma

            branch_info_list.append(template_info)
            continue

    return branch_info_list


if __name__ == '__main__':

    path = rf'..\Data\swc\mouse\mouse_registered_1um_re1_restem8_re5_reapical'
    ref_path = rf'..\Data\swc\mouse\mouse_registered_1um_re1_restem8_re5_reapical'

    fplist = glob.glob(os.path.join(path, '*.swc'))

    all_name = []
    for fp in fplist:
        fname = os.path.split(fp)[-1]
        ref_p = os.path.join(ref_path, fname)
        print(fname)
        # try:
        branch_info_list = n_branch(fp, ref_p, typelist=[1,4])
        if len(branch_info_list)==0: continue

        # except ValueError as e:
        #     print(e)

        tmplist = [list(x.values()) for x in branch_info_list]
        df = pd.DataFrame(tmplist, columns=list(branch_info_list[0].keys()))
        df['straightness'] = df['euclidean_distance'] / df['path_distance']
        df['r_l_angle_ratio'] = df['end_remote_angle'] / df['end_local_angle']
        df['name'] = fname
        df['deviation_angle'] = None

        for ind in df.index:
            curbranch = df.loc[ind]
            pid = curbranch.loc['pid']
            if pid == -1: continue
            pbranch = df.loc[pid]
            v1 = pbranch.loc[['x_end', 'y_end', 'z_end']].values - pbranch.loc[['x_start', 'y_start', 'z_start']].values
            v1 = v1 / np.linalg.norm(v1)
            v2 = curbranch.loc[['x_end', 'y_end', 'z_end']].values - curbranch.loc[
                ['x_start', 'y_start', 'z_start']].values
            v2 = v2 / np.linalg.norm(v2)
            deviation_angle = 180 * np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)) / np.pi
            df.loc[ind, 'deviation_angle'] = deviation_angle

        for ind in df.index:
            curbranch = df.loc[ind]
            pid = curbranch.loc['pid']
            accum_euc = curbranch['euclidean_distance']
            accum_path = curbranch['path_distance']
            while pid != -1:
                curbranch = df.loc[pid]
                pid = curbranch.loc['pid']
                accum_euc += curbranch['euclidean_distance']
                accum_path += curbranch['path_distance']
            df.loc[ind, 'accumulated_euc_dist'] = accum_euc
            df.loc[ind, 'accumulated_path_dist'] = accum_path

        pid_vc = df[df['pid'] != -1]['pid'].value_counts()
        df['sister_id'] = None
        df['plane_deviation_angle'] = None
        for bid in pid_vc.index[pid_vc == 2]:
            tmpdf = df[df['pid'] == bid]
            df.loc[tmpdf.index[0], 'sister_id'] = tmpdf.index[1]
            df.loc[tmpdf.index[1], 'sister_id'] = tmpdf.index[0]

            tmpbranch = df.loc[bid]
            V = tmpbranch[['x_end', 'y_end', 'z_end']].values - tmpbranch[['x_start', 'y_start', 'z_start']].values
            p_p0 = df.loc[tmpdf.index[0], ['x_start', 'y_start', 'z_start']].values.astype(np.float64)
            p_p1 = df.loc[tmpdf.index[0], ['x_end', 'y_end', 'z_end']].values.astype(np.float64)
            p_p2 = df.loc[tmpdf.index[1], ['x_end', 'y_end', 'z_end']].values.astype(np.float64)
            l10 = p_p1 - p_p0
            l20 = p_p2 - p_p0
            N = np.cross(l10, l20)
            df.loc[bid, 'plane_deviation_angle'] = np.arccos(
                np.abs(np.dot(V, N) / (np.linalg.norm(V) * np.linalg.norm(N)))) / np.pi * 180

        columns = ['id', 'branch_level', 'pid', 'sister_id',
                   'x_start', 'y_start', 'z_start', 'x_end', 'y_end', 'z_end',
                   'euclidean_distance', 'path_distance', 'straightness',
                   'deviation_angle', 'deviation_angle_soma', 'plane_deviation_angle',
                   'end_local_angle', 'end_remote_angle', 'r_l_angle_ratio',
                   'euc_dist_soma_branch_start', 'euc_dist_soma_branch_end',
                   'accumulated_euc_dist', 'accumulated_path_dist',
                   'is_terminal', 'is_cutoff', 'name']
        df = df[columns]
        df.to_csv(
            rf'..\Data\branch_info\mouse\magnify_{os.path.split(path)[-1].split("_")[-1]}_apical\{fname.replace(".swc", ".csv")}',
            index=False)
        # break
