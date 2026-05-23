import sys, os

sys.path.append(r'..')
from basicfunc import read_swc, get_soma, save_swc, sort_swc_index
import numpy as np
import pandas as pd
# import sympy
from skspatial.objects import Line, Sphere


def two_node_interp_from_soma_dist(node1, node2, soma, dist):
    node1 = np.asarray(node1)
    node2 = np.asarray(node2)
    soma = np.asarray(soma)
    x1, y1, z1 = node1[2:5] - soma[2:5]
    x2, y2, z2 = node2[2:5] - soma[2:5]
    line_segment_len = np.linalg.norm([x1 - x2, y1 - y2, z1 - z2])
    xs, ys, zs = soma[2:5]

    sphere = Sphere([0, 0, 0], dist)
    line = Line([x1, y1, z1], [x2 - x1, y2 - y1, z2 - z1])
    p1, p2 = sphere.intersect_line(line)
    l_direction = np.array([x2 - x1, y2 - y1, z2 - z1])
    l1_p1 = p1 - np.array([x1, y1, z1])
    # l2_p1 = p1 - np.array([x2, y2, z2])
    # l1_p2 = p2 - np.array([x1, y1, z1])
    # l2_p2 = p2 - np.array([x2, y2, z2])
    l1_p1_len = np.linalg.norm(l1_p1)
    # l2_p1_len = np.linalg.norm(l2_p1)
    # l1_p2_len = np.linalg.norm(l1_p2)
    # l2_p2_len = np.linalg.norm(l2_p2)
    if np.dot(l1_p1, l_direction) > 0:
        # ratio = 1 - l1_p1_len / line_segment_len
        # assert ratio <= 1, "wrong interpolation point"
        return np.asarray(p1) + soma[2:5]
    else:
        return np.asarray(p2) + soma[2:5]


def crop(swcpath, thres, magnify=None, translate=None):
    dst = []
    src = read_swc(swcpath)
    NeuronHash = {}
    indexChildren = []
    for i in range(len(src)):
        NeuronHash[src[i][0]] = i
        indexChildren.append([])
    for i in range(len(src)):
        pid = src[i][6]
        idx = NeuronHash.get(pid)
        if idx is None: continue
        indexChildren[idx].append(i)

    src_new_id = np.max(np.array(src)[:, 0]) + 1
    # DBS
    root = get_soma(src)

    if magnify is not None:
        for i in range(len(src)):
            for j in [2, 3, 4]:
                src[i][j] = root[j] + (src[i][j] - root[j]) * magnify

    root_xyz = np.array(root[2:5])
    root_n = root[0]
    root_idx = NeuronHash[root_n]
    dst.append(src[root_idx])
    bifurs = indexChildren[root_idx]
    cur_node_idx = root_idx
    while bifurs:
        next_node_idx = bifurs.pop()
        # print(cur_node_idx)
        if np.linalg.norm(np.array(src[next_node_idx][2:5]) - root_xyz) > thres:
            cur_node = src[cur_node_idx]
            p_interp = two_node_interp_from_soma_dist(cur_node, src[next_node_idx], root, thres)
            dst.append([src_new_id, cur_node[1], p_interp[0], p_interp[1], p_interp[2], cur_node[5], cur_node[0]])
            src_new_id += 1
            continue
        dst.append(src[next_node_idx])
        cur_node_idx = next_node_idx
        cur_node_child_idx = indexChildren[cur_node_idx]
        # one child
        while len(cur_node_child_idx) == 1:
            next_node_idx = cur_node_child_idx[0]
            if np.linalg.norm(np.array(src[next_node_idx][2:5]) - root_xyz) > thres:
                cur_node = src[cur_node_idx]
                p_interp = two_node_interp_from_soma_dist(cur_node, src[next_node_idx], root, thres)
                dst.append([src_new_id, cur_node[1], p_interp[0], p_interp[1], p_interp[2], cur_node[5], cur_node[0]])
                src_new_id += 1
                break
            cur_node_idx = next_node_idx
            dst.append(src[cur_node_idx])
            cur_node_child_idx = indexChildren[cur_node_idx]

        # two children or no children
        if len(cur_node_child_idx) > 1 or len(cur_node_child_idx) == 0:
            # dst.append(src[cur_node_idx])
            bifurs.extend(cur_node_child_idx)

    return dst


def crop_original(swcpath, thres):
    dst = []
    src = read_swc(swcpath)
    NeuronHash = {}
    indexChildren = []
    for i in range(len(src)):
        NeuronHash[src[i][0]] = i
        indexChildren.append([])
    for i in range(len(src)):
        pid = src[i][6]
        idx = NeuronHash.get(pid)
        if idx is None: continue
        indexChildren[idx].append(i)

    # DBS
    root = get_soma(src)
    root_xyz = np.array(root[2:5])
    root_n = root[0]
    root_idx = NeuronHash[root_n]

    dst.append(src[root_idx])
    bifurs = indexChildren[root_idx]
    while bifurs:
        cur_node_idx = bifurs.pop()
        # print(cur_node_idx)
        cur_node_child_idx = indexChildren[cur_node_idx]
        if np.linalg.norm(np.array(src[cur_node_idx][2:5]) - root_xyz) > thres:
            continue
        dst.append(src[cur_node_idx])
        # one child
        while len(cur_node_child_idx) == 1:
            next_node_idx = cur_node_child_idx[0]
            if np.linalg.norm(np.array(src[next_node_idx][2:5]) - root_xyz) > thres:
                break
            cur_node_idx = next_node_idx
            dst.append(src[cur_node_idx])
            cur_node_child_idx = indexChildren[cur_node_idx]

        # two children or no children
        if len(cur_node_child_idx) > 1 or len(cur_node_child_idx) == 0:
            # dst.append(src[cur_node_idx])
            bifurs.extend(cur_node_child_idx)

    return dst


if __name__ == "__main__":
    print()
    path = r'..\Data\swc\human\human_1um_pruned_rm_trifur_re5_reapical'

    savepathroot = r'..\Data\swc\multi_level\human'

    filelist = os.listdir(path)
    for thres in [50, 100]:
        savefolder = os.path.join(savepathroot, f"human_1um_pruned_rm_trifur_re5_reapical_crop{thres}")
        os.makedirs(savefolder, exist_ok=True)
        for filename in filelist:
            fp = os.path.join(path, filename)
            print(filename)
            # thres = 128
            dst1 = crop(fp, thres)
            dst2 = sort_swc_index(dst1)
            save_fp = os.path.join(savefolder, filename)
            save_swc(save_fp, dst2)
            # break
        # break

if __name__ == "__main1__":
    print()

    df_ct_mouse = pd.read_csv(r'..\Data\metadata\mouse_celltype.csv', index_col=0)
    df_ct_human = pd.read_csv(r'..\Data\metadata\human_celltype.csv', index_col=0)
    df_soma_d_mouse = pd.read_csv(r'..\Data\metadata\mouse_soma_diameter.csv', index_col=0)
    df_soma_d_human = pd.read_csv(r'..\Data\metadata\human_soma_diameter.csv', index_col=0)
    df_res_mouse = pd.read_excel(
        r'..\Tables\TableS1_SEU-ALLEN_brains_1223_204brains.xlsx')  # can be fetched from Liu et al., 2024; Neuronal diversity and stereotypy at multiple scales through whole brain morphometry, Supplementary Table 1.
    df_res_mouse = df_res_mouse[df_res_mouse['Research lab'] == 'U19 Zeng']
    df_res_mouse = pd.Series(df_res_mouse.iloc[:, 4].values, index=df_res_mouse['Image ID'].astype(int).astype(str))
    df_res_human = pd.Series([int(x.split('_')[4][1:]) / 1000 for x in df_ct_human.index],
                             index=[x.split('_')[0] for x in df_ct_human.index])
    for ind in df_soma_d_mouse.index:
        df_soma_d_mouse.loc[ind] = df_soma_d_mouse.loc[ind] * df_res_mouse.loc[ind.split('_')[0]]
    for ind in df_soma_d_human.index:
        df_soma_d_human.loc[ind, ['x1', 'y1', 'x2', 'y2']] = df_soma_d_human.loc[ind, ['x1', 'y1', 'x2', 'y2']] * \
                                                             df_res_human.loc[f'{ind:05}']

    h_soma_sqrt = np.sqrt((df_soma_d_human['x2'].values - df_soma_d_human['x1'].values) * \
                          (df_soma_d_human['y2'].values - df_soma_d_human['y1'].values))
    m_soma_sqrt = np.sqrt((df_soma_d_mouse['x2'].values - df_soma_d_mouse['x1'].values) * \
                          (df_soma_d_mouse['y2'].values - df_soma_d_mouse['y1'].values))

    h_soma_mean = np.mean(h_soma_sqrt)
    m_soma_mean = np.mean(m_soma_sqrt)

    soma_ratio_h_m = h_soma_mean / m_soma_mean

    print(h_soma_mean, m_soma_mean, soma_ratio_h_m)

    # crop1
    path = r'..\Data\swc\mouse\mouse_registered_1um_re1_restem8_re5_reapical'

    savepathroot = r'..\Data\swc\multi_level\mouse'

    filelist = os.listdir(path)
    for thres in [50, 10000]:
        # for thres in [43]:
        savefolder = os.path.join(savepathroot, f"mouse_registered_1um_re1_restem8_re5_magnify_crop{thres}")
        os.makedirs(savefolder, exist_ok=True)
        for filename in filelist:
            fp = os.path.join(path, filename)
            print(filename)
            # thres = 128
            dst1 = crop(fp, thres, magnify=soma_ratio_h_m)
            dst2 = sort_swc_index(dst1)
            save_fp = os.path.join(savefolder, filename)
            save_swc(save_fp, dst2)
            # break
        # break
