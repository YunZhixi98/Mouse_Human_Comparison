import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import os
import math
import warnings
from matplotlib import cm
import umap.umap_ as umap

from basicfunc import read_swc, get_soma, generate_linelist

warnings.filterwarnings("ignore")

# -------------------------------
# Helper functions
# -------------------------------

def preprocess_morpho(df, pca_frac=0.95):
    """Preprocess morphology dataframe:
       - remove near-zero variance
       - optionally remove highly correlated features
       - z-score scale
       - optionally perform PCA to keep fraction pca_frac of variance
    """
    df = df.copy()
    # remove near-zero variance
    stds = df.std(axis=0, ddof=0)
    keep = stds > 1e-8
    df = df.loc[:, keep]

    scaler = StandardScaler()
    arr = scaler.fit_transform(df.values)
    df_scaled = pd.DataFrame(arr, index=df.index, columns=df.columns)
    # PCA optional
    if pca_frac is not None and 0 < pca_frac < 1:
        pca = PCA(svd_solver='full', random_state=_GLOBAL_RANDOM_SEED)
        pcs = pca.fit_transform(df_scaled.values)
        var_exp = np.cumsum(pca.explained_variance_ratio_)
        k = np.searchsorted(var_exp, pca_frac) + 1
        df_scaled = pd.DataFrame(pcs[:, :k], index=df.index, columns=[f'PC{i+1}' for i in range(k)])
    return df_scaled

def hierarchical_clustering(dist_mat, method='ward'):
    """Perform hierarchical clustering using linkage on pairwise distance matrix (condensed)."""
    Z = linkage(dist_mat, method=method)
    return Z

def compute_wk(data, labels):
    """Compute within-cluster dispersion W(k): sum of pairwise distances / (2*n_k) per cluster, then sum log(Wk) as in Tibshirani.
       We'll compute the unnormalized Wk used in gap statistic (sum of pairwise distances within clusters).
    """
    X = np.asarray(data)
    unique = np.unique(labels)
    wk = 0.0
    for c in unique:
        members = X[labels == c]
        n = members.shape[0]
        if n <= 1:
            continue
        # sum of squared distances to cluster centroid (more stable than pairwise sum)
        centroid = members.mean(axis=0)
        ss = np.sum((members - centroid)**2)
        wk += ss
    return wk

def gap_statistic(data, ks=range(2,11), B=100, ref='uniform', random_state=None):
    """Compute gap statistic for hierarchical clustering with ward linkage.
       Returns gap values, sk (std error), Wks, and reference mean logs.
    """
    rng = np.random.RandomState(random_state)
    n, p = data.shape
    # bounding box for uniform reference
    mins = data.min(axis=0)
    maxs = data.max(axis=0)
    # compute Wk for original data
    condensed = pdist(data, metric='euclidean')
    Z = linkage(condensed, method='ward')
    Wks = []
    for k in ks:
        labels = fcluster(Z, t=k, criterion='maxclust')
        wk = compute_wk(data, labels)
        Wks.append(wk)
    Wks = np.array(Wks)
    logWks = np.log(Wks + 1e-12)
    # reference
    logWk_refs = np.zeros((len(ks), B))
    for b in range(B):
        # Generate uniform reference
        if ref == 'uniform':
            Xb = rng.uniform(mins, maxs, size=(n, p))
        else:
            # gaussian reference: sample from normal with same mean/std
            Xb = rng.normal(np.mean(data, axis=0), np.std(data, axis=0), size=(n, p))
        Zb = linkage(pdist(Xb, metric='euclidean'), method='ward')
        for i,k in enumerate(ks):
            lb = fcluster(Zb, t=k, criterion='maxclust')
            wkb = compute_wk(Xb, lb)
            logWk_refs[i, b] = np.log(wkb + 1e-12)
    logWk_ref_mean = logWk_refs.mean(axis=1)
    logWk_ref_std = logWk_refs.std(axis=1, ddof=1) * np.sqrt(1 + 1.0/B)  # sk as in original paper
    gaps = logWk_ref_mean - logWks
    return {'ks': list(ks), 'gaps': gaps, 'sk': logWk_ref_std, 'logWks': logWks, 'logWk_refs': logWk_refs}

def evaluate_k(data, Z, k_min=3, k_max=10):
    """Compute silhouette, CH, DB, WSS for range of k using hierarchical clustering result Z."""
    dist_mat = squareform(pdist(data, metric='euclidean'))
    ks = list(range(k_min, min(k_max, data.shape[0]-1) + 1))
    sil = []
    ch = []
    db = []
    wss = []
    for k in ks:
        labels = fcluster(Z, t=k, criterion='maxclust')
        # need at least 2 clusters with >=1 samples for silhouette
        try:
            s = silhouette_score(dist_mat, labels, metric='precomputed')
        except Exception:
            s = np.nan
        sil.append(s)
        try:
            chv = calinski_harabasz_score(data, labels)
        except Exception:
            chv = np.nan
        ch.append(chv)
        try:
            dbv = davies_bouldin_score(data, labels)
        except Exception:
            dbv = np.nan
        db.append(dbv)
        w = compute_wk(data, labels)
        wss.append(w)
    return pd.DataFrame({'k': ks, 'silhouette': sil, 'CH': ch, 'DB': db, 'Wk': wss})


_GLOBAL_RANDOM_SEED = 821
_USED_FEATURE = ['Center Shift', 'Average Contraction',
       'Average Bifurcation Angle Remote', 'Average Bifurcation Angle Local', 
       'Max Branch Order', 'Number of Bifurcations', 'Total Length',
       'Max Path Distance',
       'Average Euclidean Distance', 'Average Path Distance', '3D Density',
       'Volume',]

def load_data(feature_csv_root, dataset_name, den_type, crop_radius=100):
    _ = ''
    if dataset_name == 'mouse':
        _ = '_restem8'
        
    feature_class = {
        "Morphology": [
            f"{dataset_name}_{den_type}_morphology{_}_crop{crop_radius}.csv"
        ],
        "N_Branch": [
            f"{dataset_name}_{den_type}_n_branch_length{_}_crop{crop_radius}.csv",
            f"{dataset_name}_{den_type}_n_branch_num{_}_crop{crop_radius}.csv"
        ]
    }

    # get feature dataframe
    feature_df = pd.DataFrame()
    for fc1 in feature_class:
        for fc2 in feature_class[fc1]:
            tmppath = os.path.join(feature_csv_root, fc1, fc2)
            tmpdf = pd.read_csv(tmppath, index_col=0)
            if fc2.find('n_branch_length') != -1:
                tmpdf.columns = [f'{den_type} Branch Length {x}' for x in tmpdf.columns]
            elif fc2.find('n_branch_num') != -1:
                if den_type == 'apical':
                    del tmpdf['1']
                tmpdf.columns = [f'{den_type} Branch Number {x}' for x in tmpdf.columns]
            else:
                tmpdf = tmpdf[_USED_FEATURE]
                tmpdf['3D Density'] = (tmpdf['Total Length']/tmpdf['Volume']).apply(np.log10)
                tmpdf['Volume'] = (tmpdf['Volume']+1).apply(np.log10)
                tmpdf.columns = [f"{den_type} {x}" for x in tmpdf.columns]
                
            feature_df = pd.concat([feature_df, tmpdf], axis=1)
                    
    if dataset_name == 'human':
        label_df = pd.read_csv(r'..\Data\celltype\human_celltype.csv', index_col=0)
        label_df = label_df.loc[label_df['is_pyramidal']==1]
        label_df = label_df.loc[label_df['layer']=='L2/3']
    elif dataset_name == 'mouse':
        label_df = pd.read_csv(r'..\Data\celltype\mouse_celltype.csv', index_col=0)
        label_df = label_df[(label_df['is_pyramidal']==1) & (label_df['layer']=='2/3')]
    else:
        raise ValueError(f'dataset name {dataset_name} will lead to wrong Y-label.')


    feature_df.dropna(inplace=True, axis=0, how='any')
    feature_df.dropna(inplace=True, axis=1, how='all')
    feature_df = feature_df.loc[:, feature_df.sum(axis=0) != 0]

    intersect_index = feature_df.index.intersection(label_df.index)

    label_df = label_df.loc[intersect_index]
    feature_df = feature_df.loc[intersect_index]

    X = feature_df
    y = label_df
    print(f'Loaded data from {dataset_name} with shape {X.shape}.')

    return X, y


def plot_umap_clusters(
    data_scaled,
    labels,
    final_k,
    skeleton_path=None,
    random_state=_GLOBAL_RANDOM_SEED,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean',
    figsize=(12, 12),
    savefig_path=None,
    skeleton_color=False,
):
    if skeleton_color:
        savefig_path = savefig_path.replace('.png', '') + '_skeleton_color.png'
        
    X = np.asarray(data_scaled)
    labels = np.asarray(labels)
    # fit UMAP
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        n_components=2
    )
    embedding = reducer.fit_transform(X)

    # prepare colors
    unique_labels = np.unique(labels)
    cmap = cm.get_cmap('Set2')
    colors = [cmap(i % cmap.N) for i in range(len(unique_labels))]
    label_to_color = {lab: colors[i] for i, lab in enumerate(unique_labels)}
    point_colors = [label_to_color[_] for _ in labels]

    if skeleton_path is None:
        centroids = np.vstack([
            embedding[labels == lab].mean(axis=0) for lab in unique_labels
        ])
        plt.figure(figsize=figsize)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=200, c=point_colors, alpha=1, marker='.',zorder=15, linewidths=2, edgecolors='k')
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            s=120,
            c=[label_to_color[_] for _ in unique_labels],
            edgecolor='k',
            linewidth=0.8,
            marker='X'
        )
        for i, lab in enumerate(unique_labels):
            plt.text(
                centroids[i, 0],
                centroids[i, 1],
                str(lab),
                fontsize=9,
                ha='center',
                va='center',
                color='white'
            )
        plt.title(f'UMAP with cluster centroids (k={final_k})')
        plt.xlabel('UMAP1', fontsize=25)
        plt.ylabel('UMAP2', fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.show()
    else:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'aspect': 'equal'})
        plt.scatter(embedding[:, 0], embedding[:, 1], s=50, c=point_colors, alpha=1, marker='.',zorder=15, linewidths=0.5, edgecolors='k')
        for i in range(len(data_scaled))[:]:
            swcpath = os.path.join(skeleton_path, data_scaled.index[i])
            swc_plot(swcpath, color=point_colors[i], shift=embedding[i], ax=ax, skeleton_color=skeleton_color)
            
        # plt.title(f'UMAP with cluster centroids (k={final_k})')
        plt.xlabel('UMAP1', fontsize=25)
        plt.ylabel('UMAP2', fontsize=25)
        plt.tick_params(axis='both', which='major', labelsize=20)
        plt.tight_layout()
        plt.savefig(savefig_path, dpi=400, bbox_inches='tight', transparent=True)
        # plt.show()
        
    

    return embedding


def swc_plot(swc_path,color,shift,ax,skeleton_color=False):
    '''
    axis: list-like with length=2, 0,1,2 represents x,y,z
    '''
    swc = read_swc(swc_path)
    soma = get_soma(swc)
    swc_pos = np.array([np.array(x[2:5])-np.array(soma[2:5]) for x in swc])
    pca = PCA(n_components=2, svd_solver='full', random_state=_GLOBAL_RANDOM_SEED)
    swc_pos_pca = pca.fit_transform(swc_pos)
    
    apical_mask = np.array([int(x[1]) == 4 for x in swc])
    swc_pos_pca_apical = swc_pos_pca[ apical_mask, : ]  # (m,2)
    center_dir_norm = np.mean(swc_pos_pca_apical, axis=0)
    center_dir_norm = center_dir_norm / np.linalg.norm(center_dir_norm)
    # rotate to align with y-axis
    vx, vy = center_dir_norm[0], center_dir_norm[1]

    # compute rotation angle phi so rotating by phi sends v -> +y
    phi = np.arctan2(vx, vy)

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    R = np.array([[cos_phi, -sin_phi],
                    [sin_phi,  cos_phi]])  # rotation matrix

    # apply rotation to all points in PCA plane
    rotated_2d = (R @ swc_pos_pca.T).T  # (n,2)

    # ensure direction points to positive y (if mean y < 0, flip 180 deg)
    mean_after = np.mean(rotated_2d[apical_mask], axis=0)
    if mean_after[1] < 0:
        rotated_2d = -rotated_2d

    # rebuild 3D-like array with zero z for plotting in 2D context
    swc_pos_rotated = np.stack([rotated_2d[:, 0], rotated_2d[:, 1], np.zeros(rotated_2d.shape[0])], axis=1)

    
    swc = np.array(swc)
    swc[:,2:5] = swc_pos_rotated * 0.001  # plot scaling
    swc = swc.tolist()
    
    linelist = generate_linelist(swc)
    for ll in linelist:
        if not skeleton_color:
            if ll[0][1]==4:
                ax.plot(ll[:,2]+shift[0], ll[:,3]+shift[1],color='magenta',zorder=14,lw=0.5,alpha=0.5)
            else:
                ax.plot(ll[:,2]+shift[0], ll[:,3]+shift[1],color='dodgerblue',zorder=13,lw=0.5,alpha=0.7)
        else:
            ax.plot(ll[:,2]+shift[0], ll[:,3]+shift[1],color=color,zorder=13,lw=0.5,alpha=0.7)
        


    

if __name__ == "__main__":
    feature_csv_root = r'../Data/Feature'
    skeleton_path_root = r'../Data/swc/multi_level'
    save_path_root = r'../Tables/mtype_clustering'
    species = 'mouse'
    df_morpho = pd.DataFrame()
    _X1, _y1 = load_data(feature_csv_root,species,'apical',100)
    _X2, _y2 = load_data(feature_csv_root,species,'basal',100)
    _X2 = _X2[_X2.columns[~_X2.columns.str.contains('Branch')]]
    df_morpho = pd.concat([_X1, _X2], axis=1)
    # df_morpho = df_morpho[df_morpho.columns[(~df_morpho.columns.str.contains('Angle'))&(~df_morpho.columns.str.contains('Contraction'))]]
    df_morpho = df_morpho[["apical Center Shift", "apical Volume", "apical Total Length", "apical Number of Bifurcations", "basal Center Shift", "basal Volume", "basal Total Length", "basal Number of Bifurcations"]]
    df_morpho.dropna(axis=0, how='any', inplace=True)
    print(df_morpho.shape)
    # print(df_morpho.columns[:20])
    # print(df_morpho.columns[20:])

    df_morpho_scaled = preprocess_morpho(df_morpho, cor_threshold=None, pca_frac=0.99)
    print(df_morpho_scaled.shape)

    # hierarchical clustering
    Z = hierarchical_clustering(pdist(df_morpho_scaled.values, metric='euclidean'), method='ward')

    # evaluate k
    k_min = 3
    k_max = 10
    eval_df = evaluate_k(df_morpho_scaled.values, Z, k_min=k_min, k_max=k_max)

    # compute gap statistic (B small for speed in demo; increase B for real data)
    gap = gap_statistic(df_morpho_scaled.values, ks=range(k_min, k_max+1), B=120, random_state=_GLOBAL_RANDOM_SEED)

    # choose k by rank aggregation
    candidates = eval_df['k'].values
    # ranks: silhouette (higher better), CH (higher better), DB (lower better), gap (higher better)
    r_sil = (-eval_df['silhouette'].rank(method='min')).astype(int)
    r_ch  = (-eval_df['CH'].rank(method='min')).astype(int)
    r_db  = (eval_df['DB'].rank(method='min')).astype(int)
    gap_series = pd.Series(gap['gaps'], index=[int(x) for x in gap['ks']]).reindex(candidates).values
    r_gap = (-pd.Series(gap_series).rank(method='min')).astype(int).values
    rank_sum = r_sil + r_ch + r_db + r_gap
    best_k = candidates[np.argmin(rank_sum)]

    # additional rule: tibshirani gap selector
    gaps = gap['gaps']
    sk = gap['sk']
    # tibshirani rule implementation:
    chosen_gap_k = None
    ks_list = gap['ks']
    for i in range(len(ks_list)-1):
        if gaps[i] >= gaps[i+1] - sk[i+1]:
            chosen_gap_k = ks_list[i]
            break
    if chosen_gap_k is None:
        chosen_gap_k = ks_list[np.argmax(gaps)]

    # reconcile: if gap and rank_sum differ, prefer the one supported by silhouette and CH
    if chosen_gap_k != best_k:
        # check silhouette and CH agreement
        sil_best = eval_df.loc[eval_df['k']==best_k,'silhouette'].values[0]
        sil_gap = eval_df.loc[eval_df['k']==chosen_gap_k,'silhouette'].values[0]
        ch_best = eval_df.loc[eval_df['k']==best_k,'CH'].values[0]
        ch_gap = eval_df.loc[eval_df['k']==chosen_gap_k,'CH'].values[0]
        # pick the candidate with higher silhouette*CH product (simple heuristic)
        prod_best = (sil_best if not math.isnan(sil_best) else -1) * (ch_best if not math.isnan(ch_best) else -1)
        prod_gap = (sil_gap if not math.isnan(sil_gap) else -1) * (ch_gap if not math.isnan(ch_gap) else -1)
        if prod_gap > prod_best:
            final_k = chosen_gap_k
        else:
            final_k = best_k
    else:
        final_k = best_k

    # final clusters
    if species == 'mouse':
        final_k = 5
    elif species == 'human':
        final_k = 5
    labels_final = fcluster(Z, t=final_k, criterion='maxclust')
    clusters_df = pd.DataFrame({'cell': df_morpho_scaled.index, 'cluster': labels_final})
    # remove cluster 1
    # if species == 'mouse':
    #     pass
    #     clusters_df = clusters_df[clusters_df['cluster'] != 1]
    #     clusters_df.loc[clusters_df['cluster'] > 1, 'cluster'] -= 1  # re-label clusters
    
    clusters_df.to_csv(rf'{save_path_root}\{species}_mtype_clustering.csv', index=False)
    
    df_morpho_scaled = df_morpho_scaled.loc[clusters_df.cell]
    
    # Print summary
    print("Evaluation table (k, silhouette, CH, DB, Wk):")
    print(eval_df.to_string(index=False))
    print("\nGap statistic (k, gap, sk):")
    for k, g, s in zip(gap['ks'], gap['gaps'], gap['sk']):
        print(f"k={k}  gap={g:.4f}  sk={s:.4f}")
    print(f"\nRank-aggregation best_k = {best_k}, Gap-based chosen k = {chosen_gap_k}, final selected k = {final_k}")
    
    if 0:
        # Plotting results (one figure per plot as required)
        # 1) Dendrogram colored by clusters
        plt.figure(figsize=(10, 4))
        dn = dendrogram(Z, labels=df_morpho_scaled.index.tolist(), no_labels=True, count_sort='ascending')
        plt.title("Dendrogram (ward linkage)")
        plt.ylabel("Height")
        plt.tight_layout()
        plt.show()

        # 2) Metric plots
        plt.figure(figsize=(8, 4))
        plt.plot(eval_df['k'], eval_df['silhouette'], marker='o')
        plt.xlabel('k'); plt.ylabel('Average silhouette'); plt.title('Silhouette by k')
        plt.axvline(final_k, linestyle='--')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8,4))
        plt.plot(eval_df['k'], eval_df['CH'], marker='o')
        plt.xlabel('k'); plt.ylabel('Calinski-Harabasz'); plt.title('Calinski-Harabasz by k')
        plt.axvline(final_k, linestyle='--')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8,4))
        plt.plot(eval_df['k'], eval_df['DB'], marker='o')
        plt.xlabel('k'); plt.ylabel('Davies-Bouldin'); plt.title('Davies-Bouldin by k (lower better)')
        plt.axvline(final_k, linestyle='--')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8,4))
        plt.plot(gap['ks'], gap['gaps'], marker='o')
        plt.errorbar(gap['ks'], gap['gaps'], yerr=gap['sk'], fmt='none')
        plt.xlabel('k'); plt.ylabel('Gap statistic'); plt.title('Gap statistic by k')
        plt.axvline(final_k, linestyle='--')
        plt.tight_layout()
        plt.show()

        # 3) Silhouette plot for final clustering
        from sklearn.metrics import silhouette_samples
        sample_silhouette_values = silhouette_samples(squareform(pdist(df_morpho_scaled.values, metric='euclidean')), labels_final, metric='precomputed')
        y_lower = 10
        plt.figure(figsize=(6,4))
        for i in range(1, final_k+1):
            ith_sil_values = sample_silhouette_values[labels_final == i]
            ith_sil_values.sort()
            size_cluster_i = ith_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i
            plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sil_values)
            plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), fontsize=8)
            y_lower = y_upper + 10
        plt.xlabel("Silhouette coefficient values")
        plt.ylabel("Cluster")
        plt.title("Silhouette plot for final clusters")
        plt.tight_layout()
        plt.show()
    
    # UMAP plot
    plot_umap_clusters(df_morpho_scaled, clusters_df['cluster'], len(np.unique(clusters_df['cluster'])), 
                       skeleton_path=rf'{skeleton_path_root}\{species}\{"human_1um_re5_crop100" if species=="human" else "mouse_registered_1um_re1_restem8_re5_reapical_crop100"}',
                       savefig_path=rf'{save_path_root}\{species}_mtype_clustering_umap.png',
                       skeleton_color=True)
    
    plot_umap_clusters(df_morpho_scaled, clusters_df['cluster'], len(np.unique(clusters_df['cluster'])), 
                       skeleton_path=rf'{skeleton_path_root}\{species}\{"human_1um_re5_crop100" if species=="human" else "mouse_registered_1um_re1_restem8_re5_reapical_crop100"}',
                       savefig_path=rf'{save_path_root}\{species}_mtype_clustering_umap.png',
                       skeleton_color=False)

    # show head of clusters assignment
    clusters_df.head(20)
