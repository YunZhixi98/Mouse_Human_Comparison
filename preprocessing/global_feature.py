import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
import os

import utils_fe
from matplotlib import pyplot as plt

import matplotlib as mpl
import scipy

mpl.use('Agg')


class Neuron():
    def __init__(self, path, mode='t', scale=1, sep=' ', from_swc=None, prefind=False):
        self.path = path
        self.swcname = os.path.split(path)[-1]
        if from_swc is None:
            self.df_swc = self.read_swc(self.path, mode=mode, scale=scale, sep=sep)
        else:
            self.df_swc = self.preprocess_swc(from_swc, mode=mode, scale=scale)
        self.swc = self.df_swc.values.tolist()
        self.length = len(self.swc)
        if prefind:
            self.soma, self.bifurs, self.tips = [], [], []
            if self.length != 0:
                self.soma = self.get_soma()
                self.bifurs = self.get_bifurs()
                self.tips = self.get_tips()

    def read_swc(self, path, mode="t", scale=1, sep=' '):
        with open(path) as f:
            rows = f.read().splitlines()

        swc_matrix = []
        for line in rows:
            if line[0] == "#":
                continue
            elem = line.strip().split(sep)[0:7]
            swc_matrix.append(elem)

        return self.preprocess_swc(swc_matrix, mode=mode, scale=scale)

    def preprocess_swc(self, swc_matrix, mode="t", scale=1):
        df = pd.DataFrame(swc_matrix, dtype=str, columns=['id', 'type', 'x', 'y', 'z', 'radius', 'pid'])
        # df = df[df["id"] != df["pid"]]
        # df = df[df["id"].isin(df["pid"].unique())]
        if isinstance(mode, list):
            mode = [str(i) for i in mode]
        if mode == "t" or mode == 'T' or mode is None:
            pass
        elif mode == "a" or mode == "A":
            df = df[df["type"].isin(['1', '2'])]
        elif mode == "d" or mode == "D":
            df = df[df["type"].isin(['1', '3', '4'])]
        elif isinstance(mode, list) and len(mode) > 0:
            df = df[df["type"].isin(mode)]
        df = df.astype(np.float32)
        df[["id", "type", "pid"]] = df[["id", "type", "pid"]].astype(int)
        # df[["x", "y", "z", "radius"]] = df[["x", "y", "z", "radius"]].astype(np.float)

        df[["x", "y", "z"]] = df[["x", "y", "z"]] * scale
        return df

    def warning_msg(self, msg):
        print("WARNING: file {0} {1}".format(self.path, msg))

    def get_soma(self):
        df = self.df_swc.copy()
        df_soma = df[(df["type"] == 1) & (df["pid"] == -1)]
        if len(df_soma) == 0:
            df_soma = df[df["pid"] == -1]
            if len(df_soma) == 0:
                self.warning_msg("no soma detected...")
        if len(df_soma) > 1:
            self.warning_msg("multiple soma detected, select the first one as soma")

        return df_soma.values[0].tolist()

    def get_bifurs(self):
        somaid = int(self.soma[0])
        series = self.df_swc["pid"].value_counts()
        # series = series[(series > 1) & (series.index != somaid)]
        series = series[(series > 1)]
        bifur_nodes_idlist = series.index.astype(int)
        bifur_nodes = self.df_swc.loc[self.df_swc["id"].isin(bifur_nodes_idlist)]

        return bifur_nodes

    def get_tips(self):
        somapid = self.soma[6]
        ids = self.df_swc["id"]
        pids = self.df_swc["pid"]
        diff = np.setdiff1d(ids, pids)
        diff = diff[diff != somapid]
        tips = []
        for node in self.swc:
            if node[0] in diff:
                tips.append(list(node))

        return tips

    def save_swc(self, path, comments='', eswc=False):
        '''
        save swc file
        :param path:save path
        :param swc:swc list
        :param comments:some remarks in line 2 in swc file
        :return:none
        '''
        if not path.endswith(".swc"):
            path += ".swc"
        with open(path, 'w') as f:
            f.writelines('#' + comments + "\n")
            f.writelines("#n,type,x,y,z,radius,parent\n")
            for node in self.swc:
                string = ""
                for i in range(len(node)):
                    item = node[i]
                    if i in [0, 1, 6]:
                        item = int(item)
                    elif i in [2, 3, 4, 5]:
                        item = f'{item:.3f}'

                    string = string + str(item) + " "
                    if not eswc:
                        if i == 6:
                            break
                string = string.strip(" ")
                string += "\n"
                f.writelines(string)


def resample(neuron: Neuron, step):
    NT = utils_fe.NeuronTree()
    NT.readSwc_fromlist(neuron.swc)
    NT_resample = NT.resample(step)
    re_swclist = []
    for nswc in NT_resample.NeuronList:
        re_swclist.append([nswc.n, nswc.type, nswc.x, nswc.y, nswc.z, nswc.r, nswc.parent])

    return Neuron(path=neuron.path, from_swc=re_swclist)


def alignment(neuron: Neuron):
    n = utils_fe.Neuron()
    n.load_eswc(neuron.swc)
    n.normalize_neuron(ntype=list(set(n.ntype)), dir_order='zyx')
    nn = n.convert_to_swclist()

    return Neuron(path=neuron.path, from_swc=nn, prefind=True)


def generate_linelist(neuron: Neuron):
    bifur_nodes = neuron.bifurs
    bifur_nodes.append(neuron.soma)
    tips = neuron.tips
    linelist = []
    linelist_append = linelist.append
    findedlist = []
    swc = np.asarray(neuron.swc)
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

        linelist_append(np.array(templist))

    return linelist


def draw_neuron(neuron: Neuron, save_path):
    def add_scalebar(h1, h2, loc=0.9, scale=10):
        h2 = abs(h2)
        length = h1 + h2
        lw = 0.5
        plt.hlines(y=-(h2 - length / 2 * (1 - loc)), xmin=-length / 2 * loc, xmax=-length / 2 * loc + scale,
                   color='black', lw=lw)
        plt.vlines(x=-length / 2 * loc, ymin=-(h2 - length / 2 * (1 - loc) * 0.92),
                   ymax=-(h2 - length / 2 * (1 - loc) * 1.08), color='black', lw=lw)
        plt.vlines(x=-length / 2 * loc + scale, ymin=-(h2 - length / 2 * (1 - loc) * 0.92),
                   ymax=-(h2 - length / 2 * (1 - loc) * 1.08), color='black', lw=lw)
        plt.text(-length / 2 * loc + scale / 2, -(h2 - length / 2 * (1 - loc + 0.04)), "{:.0f}μm".format(scale),
                 horizontalalignment="center", verticalalignment="center", fontsize=5)

    ntypes = {"Full": [], "Axon": [1, 2], "Dendrite": [1, 3, 4]}

    try:
        df_swc = neuron.df_swc
    except FileNotFoundError:
        return False
    if len(df_swc) <= 1:
        return False

    for N_type in ntypes.keys():
        if N_type == "Full":
            swc = df_swc.values.tolist()
            tmpn = Neuron(path=neuron.path, from_swc=swc, prefind=True)
            tmpn = alignment(resample(tmpn, step=25))
            soma = tmpn.get_soma()

        elif N_type == "Axon":
            swc = df_swc[df_swc["type"].isin(ntypes.get(N_type))].values.tolist()
            tmpn = Neuron(path=neuron.path, from_swc=swc, prefind=True)
            tmpn = alignment(resample(tmpn, step=25))
            soma = tmpn.get_soma()

        else:
            swc = df_swc[df_swc["type"].isin(ntypes.get(N_type))].values.tolist()
            tmpn = Neuron(path=neuron.path, from_swc=swc, prefind=True)
            tmpn = alignment(resample(tmpn, step=10))
            soma = tmpn.get_soma()

        swc = tmpn.swc
        linelist = generate_linelist(tmpn)
        # tmpn.save_swc(r"E:\ZhixiYun\Projects\GitHub\neuron_analysis\tmp\test"+"\\"+N_type)

        ratio = 1
        for orient in ["XY", "YZ", "XZ"]:
            if orient == "XY":
                x1, x2 = 2, 3
            elif orient == "YZ":
                x1, x2 = 3, 4
            elif orient == "XZ":
                x1, x2 = 2, 4
            canvas_w1 = np.max(np.array(swc)[:, x1])
            canvas_h1 = np.max(np.array(swc)[:, x2])
            canvas_w2 = np.min(np.array(swc)[:, x1])
            canvas_h2 = np.min(np.array(swc)[:, x2])

            max_pos = np.max([canvas_w1 - canvas_w2, canvas_h1 - canvas_h2]) / 2

            min_pos = -max_pos

            plt.figure(figsize=(5, 5))
            plt.axes().set_aspect('equal', adjustable='datalim')
            for line in linelist:
                if line[0, 1] == 2:
                    color = "red"
                    pline = plt.plot(line[:, x1] * ratio, line[:, x2] * ratio, c=color, lw=0.2 * ratio)
                    pline[0].set_zorder(0)
                elif line[0, 1] == 4:
                    color = "mediumvioletred"
                    pline = plt.plot(line[:, x1] * ratio, line[:, x2] * ratio, c=color, lw=0.2 * ratio)
                    pline[0].set_zorder(1)
                else:
                    color = "dodgerblue"
                    pline = plt.plot(line[:, x1] * ratio, line[:, x2] * ratio, c=color, lw=0.2 * ratio)
                    pline[0].set_zorder(2)

            pdot = plt.scatter(soma[x1], soma[x2], c='black', s=1 * ratio)
            pdot.set_zorder(3)
            #         plt.ylim(canvas_h2*ratio,canvas_h1*ratio)
            #         plt.xlim(min_pos*ratio,max_pos*ratio)
            xlimmax = max(abs(2 * canvas_w2 / (canvas_w1 - canvas_w2) * max_pos * ratio),
                          abs(2 * canvas_w1 / (canvas_w1 - canvas_w2) * max_pos * ratio))
            plt.ylim(2 * canvas_h2 / (canvas_h1 - canvas_h2) * max_pos * ratio,
                     2 * canvas_h1 / (canvas_h1 - canvas_h2) * max_pos * ratio)
            plt.xlim(-xlimmax, xlimmax)

            scalebar_size = 0
            ori_size = 500
            while True:
                if scalebar_size == 0:
                    scalebar_size = (max_pos * 2 / 15 // ori_size) * ori_size
                    if ori_size > 300:
                        ori_size -= 100
                    elif ori_size > 100:
                        ori_size -= 50
                    elif ori_size > 10:
                        ori_size -= 10
                    else:
                        ori_size -= 1
                    if ori_size < 1:
                        ori_size = "{:.1f}".format(ori_size)
                    else:
                        ori_size = "{:.0f}".format(ori_size)
                    ori_size = float(ori_size)
                else:
                    break
            add_scalebar(canvas_h1, canvas_h2, 0.9, scalebar_size)

            plt.axis("off")

            plt.savefig(os.path.join(save_path, "Img_{0}_{1}.png".format(N_type, orient)), facecolor="white",
                        bbox_inches='tight', dpi=300)
            plt.close()

    return True


class SWC_Features():
    '''
    some new features. (see feature_name)
    (swc need resample step=10!!!)
    '''

    def __init__(self, neuron: Neuron, swc_reg=None):
        self.neuron = neuron
        self.feature_name = ["Center Shift", "Relative Center Shift",
                             "Average Contraction", "Average Bifurcation Angle Remote",
                             "Average Bifurcation Angle Local",
                             "Max Branch Order", "Number of Bifurcations", "Total Length",
                             "Max Euclidean Distance", "Max Path Distance", "Average Euclidean Distance",
                             "25% Euclidean Distance",
                             "50% Euclidean Distance", "75% Euclidean Distance", "Average Path Distance",
                             "25% Path Distance",
                             "50% Path Distance", "75% Path Distance",
                             '2D Density', '3D Density',
                             'Area', 'Volume', 'Width', 'Width_95ci', 'Height', 'Height_95ci', 'Depth', 'Depth_95ci',
                             'Slimness', 'Slimness_95ci', 'Flatness', 'Flatness_95ci']
        self.feature_dict = {}
        for fn in self.feature_name:
            self.feature_dict[fn] = None

        if self.neuron.length == 0:
            return

        self.swc = neuron.swc
        self.path = neuron.path
        self.soma = self.neuron.soma
        self.tips = self.neuron.tips
        self.swc_reg = swc_reg
        # self.bifurs = get_bifurs(swc)

        self.calc_feature()

    def calc_feature(self):
        # self.Euc_Dis(self.swc)
        self.Pat_Dis_xuan(self.swc)
        self.center_shift(self.swc)
        if self.feature_dict["Max Euclidean Distance"] is None:
            self.feature_dict["Relative Center Shift"] = None
        else:
            self.feature_dict["Relative Center Shift"] = self.feature_dict["Center Shift"] / self.feature_dict[
                "Max Euclidean Distance"]
        self.size_related_features(self.swc)
        self.xyz_approximate(self.swc)
        self.feature_dict["Number of Bifurcations"] = len(self.neuron.bifurs)

    def Euc_Dis(self, swc):
        dislist = cdist(np.array([self.soma[2:5]]), np.array(swc)[:, 2:5])[0]
        if len(dislist) == 0:
            return [None] * 5
        euc_dis_ave = np.mean(dislist)
        euc_dis_max = np.max(dislist)
        euc_dis_25, euc_dis_50, euc_dis_75 = np.percentile(dislist, [25, 50, 75])

        self.feature_dict["Max Euclidean Distance"] = euc_dis_max
        self.feature_dict["Average Euclidean Distance"] = euc_dis_ave
        self.feature_dict["25% Euclidean Distance"] = euc_dis_25
        self.feature_dict["50% Euclidean Distance"] = euc_dis_50
        self.feature_dict["75% Euclidean Distance"] = euc_dis_75

        return

    def Pat_Dis_xuan(self, swc):
        NT = utils_fe.NeuronTree()
        NT.readSwc_fromlist(swc)
        NT.computeFeature()
        pathdislist = NT.pathTotal
        euxdislist = NT.euxTotal
        length = len(pathdislist)
        if len(pathdislist) == 0 or len(euxdislist) == 0:
            return [None] * 5
        path_dis_ave = np.mean(pathdislist)
        path_dis_max = np.max(pathdislist)
        path_dis_25, path_dis_50, path_dis_75 = np.percentile(pathdislist, [25, 50, 75])

        euc_dis_ave = np.mean(euxdislist)
        euc_dis_25, euc_dis_50, euc_dis_75 = np.percentile(euxdislist, [25, 50, 75])

        self.feature_dict["Max Path Distance"] = path_dis_max
        self.feature_dict["Average Path Distance"] = path_dis_ave
        self.feature_dict["25% Path Distance"] = path_dis_25
        self.feature_dict["50% Path Distance"] = path_dis_50
        self.feature_dict["75% Path Distance"] = path_dis_75
        self.feature_dict["Total Length"] = NT.Length
        self.feature_dict["Average Contraction"] = NT.Contraction
        self.feature_dict["Average Bifurcation Angle Remote"] = NT.BifA_remote
        self.feature_dict["Average Bifurcation Angle Local"] = NT.BifA_local
        self.feature_dict["Max Branch Order"] = NT.Max_Order
        self.feature_dict["Max Euclidean Distance"] = NT.Max_Eux
        self.feature_dict["Average Euclidean Distance"] = euc_dis_ave
        self.feature_dict["25% Euclidean Distance"] = euc_dis_25
        self.feature_dict["50% Euclidean Distance"] = euc_dis_50
        self.feature_dict["75% Euclidean Distance"] = euc_dis_75

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
                    cur_dist = np.linalg.norm(np.array(self.soma[2:5]) - np.array(node[2:5]), ord=2)
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
                delta_pathdist = np.linalg.norm(np.array(cur_node[2:5]) - np.array(new_node[2:5]), ord=2)
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
            return [None] * 5
        path_dis_ave = np.mean(pathdislist)
        path_dis_max = np.max(pathdislist)
        path_dis_25, path_dis_50, path_dis_75 = np.percentile(pathdislist, [25, 50, 75])

        self.feature_dict["Max Path Distance"] = path_dis_max
        self.feature_dict["Average Path Distance"] = path_dis_ave
        self.feature_dict["25% Path Distance"] = path_dis_25
        self.feature_dict["50% Path Distance"] = path_dis_50
        self.feature_dict["75% Path Distance"] = path_dis_75

        return

    def center_shift(self, swc):
        soma = self.soma
        swc_ar = np.array(swc)
        centroid = np.mean(swc_ar[:, 2:5], axis=0)
        self.feature_dict["Center Shift"] = np.linalg.norm(np.array(soma[2:5]) - centroid[0:3], ord=2)
        return

    def pixel_voxel_calc(self, swc_xyz):
        swcxyz = np.array(swc_xyz)
        x = np.round(swcxyz[:, 0])
        y = np.round(swcxyz[:, 1])
        z = np.round(swcxyz[:, 2])
        pixels = list(set(list(zip(z, y))))  # 投射到zy平面算pixel z是主方向 且去除了冗余pixel
        voxels = list(set(list(zip(x, y, z))))
        num_pixels = len(pixels)
        num_voxels = len(voxels)

        return num_pixels, num_voxels

    def size_related_features(self, swc):
        num_nodes = len(swc)
        if num_nodes <= 3:
            return [None] * 4
        swc_zy = np.array(swc)[:, 3:5]
        swc_xyz = np.array(swc)[:, 2:5]

        try:
            CH2D = ConvexHull(swc_zy)
            CH3D = ConvexHull(swc_xyz)
        except scipy.spatial.qhull.QhullError:
            return [None] * 4
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
                count = int(np.linalg.norm([x1 - x2, y1 - y2, z1 - z2]) // 1)
                if count != 0:
                    tmp = [[x1 + 1 * x, y1 + 1 * x, z1 + 1 * x] for x in
                           range(1, count + 1)]
                    swc_xyz_new.extend(tmp)

        num_pixels, num_voxels = self.pixel_voxel_calc(swc_xyz_new)
        density_2d = num_pixels / area
        density_3d = num_voxels / volume
        self.feature_dict["Area"] = area
        self.feature_dict["Volume"] = volume
        self.feature_dict["2D Density"] = density_2d
        self.feature_dict["3D Density"] = density_3d

        return

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

        self.feature_dict["Width"] = width
        self.feature_dict["Height"] = height
        self.feature_dict["Depth"] = depth
        self.feature_dict["Width_95ci"] = width_95ci
        self.feature_dict["Height_95ci"] = height_95ci
        self.feature_dict["Depth_95ci"] = depth_95ci
        self.feature_dict["Slimness"] = slimness
        self.feature_dict["Flatness"] = flatness
        self.feature_dict["Slimness_95ci"] = slimness_95ci
        self.feature_dict["Flatness_95ci"] = flatness_95ci

        return


def Pipeline(file_path, scale=1, mode=None):
    fp_split = os.path.split(file_path)
    folder_path = fp_split[0]
    swcname = fp_split[-1]

    swcpath = os.path.join(folder_path, swcname)

    n_all = Neuron(swcpath, scale=scale, mode=mode)
    # draw_neuron(n_all, save_path=folder_path)

    df_swc = n_all.df_swc

    # swc_axon = df_swc[df_swc["type"].isin([1, 2])].values.tolist()  # axon+den
    swc_den = df_swc.values.tolist()

    # df_swc["type"]=3    # only den
    # df_swc["type"][df_swc["pid"]==-1]=1
    # swc_den = df_swc.values.tolist()

    # df_swc["type"] = 2  # only axon
    # df_swc["type"][df_swc["pid"] == -1] = 1
    # swc_axon = df_swc.values.tolist()

    # n_axon = Neuron(swcpath, from_swc=swc_axon)
    n_den = Neuron(swcpath, from_swc=swc_den, prefind=False)

    # n_axon_re10 = resample(n_axon, step=10)
    # n_den_re4 = resample(n_den, step=4)

    # n_axon_re10_align = alignment(n_axon_re10)
    n_den_re4_align = alignment(n_den)
    # n_den_re4_align.save_swc(savepath, comments='neuron alignment')
    # n_axon_re10_align_fe = SWC_Features(n_axon_re10_align)
    n_den_re4_align_fe = SWC_Features(n_den_re4_align)

    return n_den_re4_align_fe


if __name__ == "__main__":
    '''
    mouse human
    '''
    for thresh in [50,100]:
    # for thresh in [100]:
        path = rf"E:\ZhixiYun\Projects\fMOST_atlas\Data\swc\multi_level\mouse\mouse_registered_1um_re1_restem8_re5_reapical_crop{thresh}"
        filelist = os.listdir(path)
        avl = []
        bvl = []
        for i in range(len(filelist)):
            filename = filelist[i]
            print(i + 1, filename, end='\t')
            filepath = os.path.join(path, filename)
            # b = Pipeline(filepath, mode=[1, 4])
            b = Pipeline(filepath)
            # av = list(a.feature_dict.values())
            bv = list(b.feature_dict.values())

            # avl.append(av)
            bvl.append(bv)
            print('finished <<< ')
        # dfa = pd.DataFrame(avl, index=filelist, columns=list(a.feature_dict.keys()))
        dfb = pd.DataFrame(bvl, index=filelist[:len(bvl)], columns=list(b.feature_dict.keys()))


        dfb.to_csv(fr"..\Data\Morphology\mouse_apical_morphology_restem8_crop{thresh}.csv")