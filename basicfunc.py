from typing import Union, List, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import json
import scipy
import statsmodels.formula.api as smf


def read_swc(path, mode="t", scale=None, comments=False):
    '''

    :param path: file path
    :param mode: "a"--Axon  "t"--total(all of it)  "d"--Dendrite
    :return: a list like [
                          [id type x y z radius pid],
                          [id type x y z radius pid],
                          ...
                          [id type x y z radius pid],
                                                     ]
    '''
    if isinstance(mode, list):
        mode = [str(i) for i in mode]
        
    swc_matrix = []
    comments_list = []
    with open(path) as f:
        while True:
            linelist = []
            line = f.readline()
            if not line:
                break
            if line[0].isalpha() or line[0] == "#" or line[0] == "\n":
                comments_list.append(line)
                continue
            if line.count("\t") >= line.count(" "):
                str_split = "\t"
            elif line.count("\t") <= line.count(" "):
                str_split = " "
            elem = line.strip("\n").strip(" ").split(str_split)
            if mode == "t" or mode == 'T':
                pass
            elif mode == "a" or mode == "A":  # 1s 2a 3d
                if elem[1] not in ['1', '2']:
                    continue
            elif mode == "d" or mode == "D":
                if elem[1] not in ['1', '3', '4']:
                    continue
            elif isinstance(mode, list) and len(mode) > 0:
                if elem[1] not in mode:
                    continue
            for i in range(len(elem)):
                if i == 0 or i == 1 or i == 6:
                    linelist.append(int(elem[i]))
                elif i in [2, 3, 4]:
                    if scale is not None:
                        linelist.append(float(elem[i]) * scale)
                    else:
                        linelist.append(float(elem[i]))
                else:
                    linelist.append(float(elem[i]))

            swc_matrix.append(linelist)

    if mode == 'bifur':
        bifur_matrix = []
        for i in range(len(swc_matrix)):
            count = 0
            for j in range(len(swc_matrix)):
                if swc_matrix[i][0] == swc_matrix[j][6]:
                    count += 1
                if count >= 2:
                    bifur_matrix.append(swc_matrix[i])
                    break
        return bifur_matrix

    if comments:
        return swc_matrix, comments_list
    else:
        return swc_matrix


def get_soma(swc):
    '''
    获取soma点
    :param swc: read_swc函数的返回值
    :return:[id type x y z radius pid]
    '''
    soma = []
    for i in range(len(swc)):
        if swc[i][1] == 1 and swc[i][6] == -1:
            soma = swc[i]
            break
    if len(soma) == 0:
        for i in range(len(swc)):
            if swc[i][6] == -1:
                soma = swc[i]
                break
    if len(soma) == 0:
        for i in range(len(swc)):
            if swc[i][1] == 1:
                soma = swc[i]
                break
    if len(soma) == 0:
        print("no soma detected...")
    return soma


def get_bifurs(swc):
    '''
    '''
    soma = get_soma(swc)
    if not soma:
        return []
    bifur_nodes = []
    df = pd.DataFrame(swc)
    df_vc = df.iloc[:, 6].value_counts()
    for i in range(len(df_vc.index)):
        if df_vc.iloc[i] == 2 and df_vc.index[i] != soma[0] and df_vc.index[i] != -1:
            try:
                idx = np.array(swc)[:, 0].tolist().index((df_vc.index[i]))
            except:
                continue
            bifur_nodes.append(swc[idx])

    return bifur_nodes


def get_tips(swc):
    tips = []
    swc = np.asarray(swc)
    tips = swc[np.in1d(swc[:, 0], np.setdiff1d(swc[:, 0], swc[:, 6]))].tolist()
    # for i in range(swc.shape[0]):
    #     if swc[i][0] not in swc[:, 6].tolist():
    #         tips.append(swc[i].tolist())
    return tips


def sort_swc_index(src):
    dst = []
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

    LUT_n2newn = {}
    count = 1

    # DBS
    root = get_soma(src)
    root_xyz = np.array(root[2:5])
    root_n = root[0]
    root_idx = NeuronHash[root_n]

    LUT_n2newn[root_n] = count
    tmpnode = list(root)
    tmpnode[0] = count
    count += 1
    dst.append(tmpnode)

    bifurs = indexChildren[root_idx]
    while bifurs:
        cur_node_idx = bifurs.pop()
        LUT_n2newn[src[cur_node_idx][0]] = count
        tmpnode = list(src[cur_node_idx])
        tmpnode[0] = count
        count += 1
        dst.append(tmpnode)
        # print(cur_node_idx)
        cur_node_child_idx = indexChildren[cur_node_idx]
        # one child
        while len(cur_node_child_idx) == 1:
            next_node_idx = cur_node_child_idx[0]
            cur_node_idx = next_node_idx
            LUT_n2newn[src[cur_node_idx][0]] = count
            tmpnode = list(src[cur_node_idx])
            tmpnode[0] = count
            count += 1
            dst.append(tmpnode)

            cur_node_child_idx = indexChildren[cur_node_idx]

        # two children or no children
        if len(cur_node_child_idx) > 1 or len(cur_node_child_idx) == 0:
            bifurs.extend(cur_node_child_idx)

    for i in range(len(dst)):
        node = dst[i]
        mapped_pid = LUT_n2newn.get(node[6])
        if mapped_pid is None:
            mapped_pid = -1
        dst[i][6] = mapped_pid

    return dst


def generate_linelist(swc):
    bifur_nodes = get_bifurs(swc)
    bifur_nodes.append(get_soma(swc))
    tips = get_tips(swc)
    linelist = []
    linelist_append = linelist.append
    findedlist = []
    swc = np.asarray(swc)
    idlist = swc[:, 0].tolist()
    bifur_nodes_idlist = np.array(bifur_nodes)[:, 0]

    for i, tempnode in enumerate(tips):
        templist = []
        templist_append = templist.append
        templist_append(tempnode)
        while True:
            if tempnode[6] in idlist:
                idx = idlist.index(tempnode[6])
            else:
                break
            tempnode = swc[idx]
            templist_append(tempnode)

            _id = tempnode[0]

            if not findedlist:
                if _id in bifur_nodes_idlist:
                    findedlist.append(tempnode)
                continue

            if _id in bifur_nodes_idlist:
                if not findedlist:
                    findedlist.append(tempnode)
                else:
                    findedlist_idlist = np.array(findedlist)[:, 0]
                    if _id not in findedlist_idlist:
                        findedlist.append(tempnode)
                    else:
                        break
            else:
                continue
        # print(i+1, "/", len(tips))

        linelist_append(np.array(templist))

    return linelist


def save_swc(path, swc, comments='', eswc=False):
    '''
    save swc file
    :param path:save path
    :param swc:swc list
    :param comments:some remarks in line 2 in swc file
    :return:none
    '''
    if not path.endswith(".swc") and not path.endswith(".eswc"):
        if eswc:
            path += ".eswc"
        else:
            path += ".swc"
    with open(path, 'w') as f:
        f.writelines('#' + comments + "\n")
        f.writelines("#n,type,x,y,z,radius,parent\n")
        for node in swc:
            string = ""
            for i in range(len(node)):
                item = node[i]
                if i in [0, 1, 6]:
                    item = int(item)
                elif i in [2, 3, 4]:
                    item = np.round(item, 3)
                string = string + str(item) + " "
                if not eswc:
                    if i == 6:
                        break
            string = string.strip(" ")
            string += "\n"
            f.writelines(string)


def stat_test(a,b,stat):
    if len(a)==0 or len(b)==0:
        return np.nan, ""

    if stat=='mwu':
        pv = scipy.stats.mannwhitneyu(a,b,alternative='two-sided')[1]
    elif stat=='ks':
        pv = scipy.stats.ks_2samp(a,b,alternative='two-sided')[1]
    elif stat=='t':
        pv = scipy.stats.ttest_rel(a,b,alternative='two-sided')[1]
    else:
        return ValueError(f'unknown statistical test method: {stat}')
        
    significance = "n.s."
  
    if pv<0.0001:
        significance = '***'
    elif pv<0.001:
        significance = '**'  
    # elif pv<0.01:
    #     significance = '**'
    elif pv<0.05:
        significance = '*'
        
    return pv,significance


def cohens_d(a, b):
    n1,n2 = len(a), len(b)
    pooled_std = np.sqrt(((n1-1)*np.std(a, ddof=1)**2 + (n2-1)*np.std(b, ddof=1)**2) / (n1+n2-2))
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_std
    return cohens_d


def cliffs_delta(x1: Union[np.ndarray, List], 
                 x2: Union[np.ndarray, List]) -> Tuple[float, str]:
    """
    Cliff's Delta - Nonparametric effect size (robust to outliers).

    Returns
    -------
    delta : float, [-1, 1]
    interpretation : str
    """
    x1, x2 = np.asarray(x1), np.asarray(x2)
    x1, x2 = x1[~np.isnan(x1)], x2[~np.isnan(x2)]

    if len(x1) == 0 or len(x2) == 0:
        return np.nan, "undefined"

    comparisons = np.sum(x1[:, None] > x2) - np.sum(x1[:, None] < x2)
    delta = comparisons / (len(x1) * len(x2))

    abs_d = abs(delta)
    if abs_d < 0.147:
        interp = "negligible"
    elif abs_d < 0.33:
        interp = "small"
    elif abs_d < 0.474:
        interp = "medium"
    else:
        interp = "large"

    return delta, interp

    
def bootstrap_stat_test(group1, group2, stat, n_iterations=1000, sample_size=1000, group1_meta=None, group2_meta=None):
        
    df1 = group1_meta.copy()
    df1['metric'] = group1
    df1['group'] = 'group1'
    # df1['weight'] = df1['brain_region'].map(df1['brain_region'].value_counts(normalize=True).rdiv(1))

    df2 = group2_meta.copy()
    df2['metric'] = group2
    df2['group'] = 'group2'
    # df2['weight'] = df2['brain_region'].map(df2['brain_region'].value_counts(normalize=True).rdiv(1))
    df = pd.concat([df1, df2], axis=0)
        
    bootstrapped_p_values = []
    # bootstrapped_p_values_mlm = []
    # cohens_d_values = []
    min_sample_size = min(len(group1),len(group2))
    if sample_size is None or sample_size > min_sample_size:
        sample_size = min_sample_size
    # print('sample_size=',sample_size)
    random_seed_list = np.arange(1,n_iterations+1,1)
    for _ in range(n_iterations):
        np.random.seed(random_seed_list[_])
        subsample_indices1 = np.random.choice(len(group1), size=sample_size, replace=True)
        subsample_indices2 = np.random.choice(len(group2), size=sample_size, replace=True)
        subsample_group1 = group1[subsample_indices1]
        subsample_group2 = group2[subsample_indices2]
        
        p_value = stat_test(subsample_group1, subsample_group2, stat)[0]
        # cohens_d_value = cohens_d(subsample_group1, subsample_group2)
        # cohens_d_values.append(cohens_d_value)
        bootstrapped_p_values.append(p_value)
    
    df['metric'] = (df['metric'] - df['metric'].mean()) / df['metric'].std()
    model = smf.mixedlm('metric ~ group', data=df, groups='subject')
    try:
        result = model.fit(reml=False,optimizer='bfgs',maxiter=5000,tol=1e-6,full_output=True)
        pv_mlm = result.pvalues['group[T.group2]']
    except:
        pv_mlm = np.nan
    if np.isnan(pv_mlm):
        try:
            print('bfgs not converged, try powell')
            result = model.fit(reml=False,optimizer='powell',maxiter=5000,tol=1e-6,full_output=True)
            pv_mlm = result.pvalues['group[T.group2]']
        except:
            print('no convergence')
            pv_mlm = np.nan
    print(pv_mlm)
    pv = np.mean(bootstrapped_p_values)
    # pv_mlm = np.mean(bootstrapped_p_values_mlm)
    # mean_cohens_d = np.mean(cohens_d_values)
    mean_cohens_d = cohens_d(group1, group2)
    significance = "n.s."
    if np.isnan(pv_mlm):
        significance = ''
    else:
        if pv_mlm<0.0001:
            significance = '***'
        elif pv_mlm<0.001:
            significance = '**'  
        # elif pv<0.01:
        #     significance = '**'
        elif pv_mlm<0.05:
            significance = '*'

    return pv, significance, mean_cohens_d, pv_mlm


def js_divergence(arr1, arr2):
    '''
    multivariate gaussian distribution based JS divergence.
    '''
    
    def kl_divergence(mu1, cov1, mu2, cov2):
        # Ensure the covariance matrices are invertible
        cov2_inv = np.linalg.inv(cov2)
        
        # Dimensionality of the data
        k = mu1.shape[0]
        
        # Compute the KL divergence
        kl_div = 0.5 * (np.log(np.linalg.det(cov2) / np.linalg.det(cov1))
                        - k
                        + np.trace(cov2_inv @ cov1)
                        + (mu2 - mu1).T @ cov2_inv @ (mu2 - mu1))
        
        return kl_div

    # Compute means and covariances for both distributions
    mu1 = np.mean(arr1, axis=0)
    mu2 = np.mean(arr2, axis=0)
    cov1 = np.cov(arr1.T)
    cov2 = np.cov(arr2.T)

    # Compute the midpoint distribution's parameters
    mu_m = 0.5 * (mu1 + mu2)
    cov_m = 0.5 * (cov1 + cov2)

    # Calculate the JS divergence
    js_div = 0.5 * kl_divergence(mu1, cov1, mu_m, cov_m) + 0.5 * kl_divergence(mu2, cov2, mu_m, cov_m)
    
    return js_div
    

def Euc_calc(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)


class SWC_Features:
    '''
    some new features. (see feature_name)
    (swc need resample)
    '''

    def __init__(self, swc, swcname, swc_reg=None):
        self.feature_name = ["Average Euclidean Distance", "25% Euclidean Distance", "50% Euclidean Distance",
                             "75% Euclidean Distance", "Average Path Distance", "25% Path Distance",
                             "50% Path Distance", "75% Path Distance", "Center Shift", "Relative Center Shift"]

        self.features = []
        self.swc = swc
        self.swcname = swcname
        self.soma = get_soma(swc)
        self.tips = get_tips(swc)
        self.swc_reg = swc_reg
        # self.bifurs = get_bifurs(swc)

        if not self.soma:
            print(swcname, "no soma detected...")
            self.features = []
        else:
            self.features += self.Euc_Dis(swc)
            self.features += self.Pat_Dis(swc)
            cs = self.center_shift(swc)
            ave_euc_dis = self.features[0]
            self.features += [cs, cs / ave_euc_dis]
            if self.swc_reg is not None:
                self.features += self.size_related_features(swc_reg)
                self.features += self.xyz_approximate(swc_reg)
                self.feature_name += [
                    "Area", 'Volume', "2D Density", "3D Density", "Width", "Height", "Depth", "Width_95ci",
                    "Height_95ci",
                    "Depth_95ci", "Slimness", "Flatness", "Slimness_95ci", "Flatness_95ci"]

    def Euc_Dis(self, swc):
        dislist = []
        soma = self.soma
        for node in swc:
            cur_dis = Euc_calc(soma[2], soma[3], soma[4], node[2], node[3], node[4])
            dislist.append(cur_dis)
        length = len(dislist)
        if length == 0:
            return [None] * 4
        euc_dis_ave = np.mean(dislist)
        dislist.sort()
        euc_dis_25 = dislist[int(np.floor(length * 1 / 4))]
        euc_dis_50 = np.median(dislist)
        euc_dis_75 = dislist[int(np.floor(length * 3 / 4))]
        return [euc_dis_ave, euc_dis_25, euc_dis_50, euc_dis_75]

    def Pat_Dis(self, swc):
        patlist = []
        soma = self.soma
        id_pathdist = {}
        idlist = np.array(swc)[:, 0].tolist()
        pidlist = np.array(swc)[:, 6].tolist()
        if soma[0] not in pidlist:
            maxdist = 1000000
            for node in swc:
                if node == self.soma:
                    continue
                if node[6] not in idlist:
                    cur_dist = Euc_calc(self.soma[2], self.soma[3], self.soma[4], node[2], node[3], node[4])
                    if cur_dist < maxdist:
                        maxdist = cur_dist
                        soma = node

        if self.tips:
            nodes = self.tips
        else:
            nodes = swc
        for node in nodes:
            if node == soma or node == self.soma:
                continue
            cur_node = node
            cur_pathdist = 0
            passbynode = {}
            while True:
                pid = cur_node[6]
                if pid not in idlist:
                    break
                idx = idlist.index(pid)
                new_node = swc[idx]
                delta_pathdist = Euc_calc(cur_node[2], cur_node[3], cur_node[4], new_node[2], new_node[3], new_node[4])
                cur_pathdist += delta_pathdist
                if passbynode.keys():
                    passbynode = dict(
                        zip(list(passbynode.keys()), (np.array(list(passbynode.values())) + delta_pathdist).tolist()))
                if new_node == soma:
                    id_pathdist[node[0]] = cur_pathdist
                    id_pathdist.update(passbynode)
                    break
                elif new_node[0] in id_pathdist.keys():
                    id_pathdist[node[0]] = cur_pathdist + id_pathdist[new_node[0]]
                    if passbynode.keys():
                        passbynode = dict(
                            zip(list(passbynode.keys()),
                                (np.array(list(passbynode.values())) + id_pathdist[new_node[0]]).tolist()))
                    id_pathdist.update(passbynode)
                    break
                else:
                    cur_node = new_node
                    passbynode[new_node[0]] = 0

        pathdislist = list(id_pathdist.values())
        length = len(pathdislist)
        if length == 0:
            return [None] * 4
        path_dis_ave = np.mean(pathdislist)
        pathdislist.sort()
        path_dis_25 = pathdislist[int(np.floor(length * 1 / 4))]
        path_dis_50 = np.median(pathdislist)
        path_dis_75 = pathdislist[int(np.floor(length * 3 / 4))]
        return [path_dis_ave, path_dis_25, path_dis_50, path_dis_75]

    def center_shift(self, swc):
        soma = self.soma
        swc_ar = np.array(swc)
        centroid = np.mean(swc_ar[:, 2:5], axis=0)
        return np.linalg.norm(soma[2:5] - centroid[0:3])

    def pixel_voxel_calc(self, swc_xyz):
        swcxyz = np.array(swc_xyz)
        x = np.round(swcxyz[:, 0])
        y = np.round(swcxyz[:, 1])
        z = np.round(swcxyz[:, 2])
        pixels = list(set(list(zip(z, y))))
        voxels = list(set(list(zip(x, y, z))))
        num_pixels = len(pixels)
        num_voxels = len(voxels)

        return num_pixels, num_voxels

    def size_related_features(self, swc):
        num_nodes = len(swc)
        if num_nodes < 3:
            return [None] * 4
        swc_zy = np.array(swc)[:, 3:5]
        swc_xyz = np.array(swc)[:, 2:5]
        CH2D = ConvexHull(swc_zy)
        CH3D = ConvexHull(swc_xyz)
        area = CH2D.volume
        volume = CH3D.volume
        # interpolation of swc so that each pixel/voxel can be occupied on all pathway
        swc_xyz_new = list(swc_xyz)
        swc_arr = np.array(swc)
        idlist = list(swc_arr[:, 0])
        for node in swc:
            pid = node[6]
            x1, y1, z1 = node[2:5]
            if pid not in idlist:
                continue
            else:
                cur_id = idlist.index(pid)
                x2, y2, z2 = swc_xyz[cur_id]
                count = int(Euc_calc(x1, y1, z1, x2, y2, z2) // 1)
                if count != 0:
                    tmp = [[x1 + 1 * x, y1 + 1 * x, z1 + 1 * x] for x in
                           range(1, count + 1)]
                    swc_xyz_new.extend(tmp)

        num_pixels, num_voxels = self.pixel_voxel_calc(swc_xyz_new)
        density_2d = num_pixels / area
        density_3d = num_voxels / volume

        return [area, volume, density_2d, density_3d]

    def xyz_approximate(self, swc):
        '''
        shape related
        :param swc:
        :return:
        '''
        if not swc:
            return [None] * 10
        swcxyz = np.array(swc)[:, 2:5]
        x = swcxyz[:, 0]
        y = swcxyz[:, 1]
        z = swcxyz[:, 2]
        width = np.max(y) - np.min(y)  # y  zyx-registration   height=z-z' width=y-y' depth=x-x'
        height = np.max(z) - np.min(z)  # z
        depth = np.max(x) - np.min(x)  # x
        # confidence interval 95%
        width_95ci = abs(np.percentile(y, 97.5) - np.percentile(y, 2.5))
        height_95ci = abs(np.percentile(z, 97.5) - np.percentile(z, 2.5))
        depth_95ci = abs(np.percentile(x, 97.5) - np.percentile(x, 2.5))

        slimness = width / height  # slimness = width/height
        flatness = height / depth  # flatness = height/depth
        slimness_95ci = width_95ci / height_95ci
        flatness_95ci = height_95ci / depth_95ci

        return [width, height, depth, width_95ci, height_95ci, depth_95ci, slimness, flatness, slimness_95ci,
                flatness_95ci]


class MouseAnatomyTree:
    # tree.json is from Allen CCF annotation ontology
    def __init__(self, treepath=r"./tree.json", ):
        self.tree = []
        self.roughlist = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'STR', 'PAL',
                          'TH', 'HY', 'MB', 'P', 'MY', 'CBX', 'CBN', 'VS', 'fiber tracts']
        self.lutnametoid = {}
        self.lutidtoname = {}
        self.lutidtorgb = {}
        self.lutnametorough = {}

        self._id_index_hash = {}

        with open(treepath) as f:
            self.tree = json.load(f)

        for i,t in enumerate(self.tree):
            id_ = t["id"]
            self.lutnametoid[t["acronym"]] = id_
            self.lutidtoname[id_] = t["acronym"]
            self.lutidtorgb[id_] = t["rgb_triplet"]
            self._id_index_hash[id_] = i

        for t in self.tree:
            self.lutnametorough[t['acronym']] = t['acronym']
            for rough in self.roughlist:
                if self.lutnametoid.get(rough) in t['structure_id_path']:
                    self.lutnametorough[t['acronym']] = rough
                    break

    def _id_acronym_check(self, inp, inp_type: str):
        if inp_type not in ['id','acronym']:
            raise ValueError(f'invalid input type: {inp_type}, should be one of "id", "acronym".')
        if inp_type == 'id':
            if isinstance(inp, str):
                inp = self.lutnametoid.get(inp)
        elif inp_type == 'acronym':
            if isinstance(inp, int):
                inp = self.lutidtoname.get(inp)
        return inp

    def _ctlist_overlap_check(self, ctlist) -> bool:
        # check ctlist whether it has overlapping cell types in tree hireachy
        ct_child_list = []
        for ct in ctlist:
            ct = self._id_acronym_check(ct,'id')
            children = self.find_children_id(ct)

            if ct in ct_child_list:
                return True

            ct_child_list.extend(children)

        return False

    def find_children_id(self, id_):
        acronym = self._id_acronym_check(id_, 'acronym')
        id_ = self._id_acronym_check(id_, 'id')
        idlist = []
        if acronym in ['SSp1','SSp2/3','SSp4','SSp5','SSp6a','SSp6b']:
            layer = acronym.replace('SSp', '')
            SSp_id = self.lutnametoid['SSp']
            for t in self.tree:
                if SSp_id in t["structure_id_path"]:
                    if t['acronym'].endswith(layer):
                        idlist.append(t['id'])
        else:
            for t in self.tree:
                if id_ in t["structure_id_path"]:
                    idlist.append(t['id'])
            
        if not idlist:
            idlist = [id_]
        return idlist

    def ccf_sort(self, ctlist):
        select_ct_sorted = []
        for item in self.tree:
            if item["acronym"] in ctlist:
                select_ct_sorted.append(item["acronym"])
        return select_ct_sorted

    def cortex_layer_to_upper(self, ctlist, SSp=False):
        newctlist = []
        _SSp_id = self.lutnametoid['SSp']
        for ct in ctlist:
            ct = self._id_acronym_check(ct,'acronym')
            if self.lutnametorough.get(ct) == 'Isocortex':
                id_ = self.lutnametoid.get(ct)
                upper_id_list = self.tree[self._id_index_hash[id_]]['structure_id_path']
                if len(upper_id_list)>=2:
                    upper_id = upper_id_list[-2]
                else:
                    upper_id = upper_id_list[-1]
                upper_acronym = self.lutidtoname.get(upper_id)

                if SSp:
                    if _SSp_id in upper_id_list:
                        upper_acronym = 'SSp'
                        newctlist.append(upper_acronym)
                        continue

                if ct[len(upper_acronym):] in ['1','2/3','4','5','6','6a','6b']:
                    newctlist.append(upper_acronym)
                else:
                    newctlist.append(ct)
            else:
                newctlist.append(ct)

        return newctlist

    def ctlist_to_given_ctlist(self, ctlist, given_ctlist, not_in_set_None=False):
        # bullshit code need to re-write
        overlap_flag = self._ctlist_overlap_check(given_ctlist)
        if overlap_flag:
            raise ValueError('given_ctlist has overlapping cell types')
        ctlist = np.asarray(ctlist)
        given_ctlist = np.asarray(given_ctlist)
        tmp_ctlist = []
        tmp_given_ctlist = []
        for ct in ctlist:
            ct = self._id_acronym_check(ct, 'id')
            tmp_ctlist.append(ct)
        tmp_ctlist = np.asarray(tmp_ctlist)

        out_arr = np.zeros(len(tmp_ctlist),dtype=object)
        if not_in_set_None:
            out_arr[out_arr==0] = None
        else:
            out_arr = np.asarray(tmp_ctlist,dtype=object)

        for gct in given_ctlist:
            gct_children = self.find_children_id(gct)
            out_arr[np.isin(tmp_ctlist, gct_children)] = gct

        for i in range(len(out_arr)):
            out_arr[i] = self._id_acronym_check(out_arr[i],'acronym')

        return out_arr



