import os
import re
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scmorph as sm
from natsort import natsort_keygen
from scipy.spatial.distance import mahalanobis
from scipy.stats import kruskal, kstest, median_abs_deviation, wasserstein_distance
from statsmodels.distributions.empirical_distribution import ECDF
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # fix file lock error


def embed(adata):
    sm.pp.scale(adata)
    sm.pp.pca(adata)


def get_adata_pca_data(adata, control, key):
    ctrl = _get_adata_subset(adata, control, key)
    ctrlcoord = _get_pca(ctrl)
    covinv = _compute_inv_cov(ctrlcoord)
    centroid = _get_centroid(ctrl)
    return ctrlcoord, covinv, centroid


def _get_adata_subset(
    adata, target="control_transfection_reagent_only", key="Treatment"
):
    if isinstance(target, str):
        assert isinstance(key, str), "key must be a string if target is a string"
        idx = adata.obs[key] == target
        return adata[idx, :]

    if isinstance(target, list):
        target = tuple(target)

    if isinstance(target, tuple):
        assert isinstance(key, list), "key must be a list if target is a tuple or list"
        assert len(target) == len(key), "target and key must have the same length"
        idx = adata.obs.groupby(key).get_group(target).index
        return adata[idx, :]

    raise ValueError("target must be a string, tuple, or list")


def _get_pca(adata):
    assert "X_pca" in adata.obsm_keys(), "run embed first"
    return adata.obsm["X_pca"] * adata.uns["pca"]["variance_ratio"]


def _get_centroid(adata):
    coord = _get_pca(adata)
    return np.mean(coord, axis=0)


def _compute_inv_cov(coord):
    cov = np.cov(coord, rowvar=False)
    covinv = np.linalg.inv(cov)
    return covinv


def compute_single_cell_dists(adata, covinv=None, centroid=None, npcs=10):
    if "X_pca" not in adata.obsm_keys():
        embed(adata)

    if centroid is None:
        centroid = _get_centroid(adata)
    coord = _get_pca(adata)

    if covinv is None:
        covinv = _compute_inv_cov(coord)

    centroid = centroid[:npcs]
    coord = coord[:, :npcs]
    covinv = covinv[:npcs, :npcs]

    dists = np.apply_along_axis(
        lambda x: mahalanobis(centroid, x, covinv), axis=1, arr=coord
    )
    return dists, covinv


def compute_single_cell_dists_per_group(
    adata, centroid, covinv, key="Treatment", group_key=None, npcs=10
):
    assert isinstance(key, str), "key must be a string"
    iterby = key
    if group_key is not None:
        if isinstance(group_key, str):
            group_key = [group_key]

        iterby = [key, *group_key]

    if isinstance(iterby, list) and len(iterby) == 1:
        iterby = iterby[0]

    out = {}
    for g, idx in adata.obs.groupby(iterby, observed=True).groups.items():
        group_adata = adata[idx]
        dist = compute_single_cell_dists(
            group_adata, covinv=covinv, centroid=centroid, npcs=npcs
        )[0]
        dist.sort()
        out[g] = dist
    return out


def distance_distributions(
    adata,
    key="Treatment",
    control="control_transfection_reagent_only",
    group_key=None,
    npcs=10,
):
    _, covinv, centroid = get_adata_pca_data(adata, control, key)

    dists = compute_single_cell_dists_per_group(
        adata, centroid, covinv, key=key, group_key=group_key, npcs=npcs
    )

    # fixup DMSO keys is group_key present
    if group_key is not None:
        ctrldist = np.concatenate([v for k, v in dists.items() if k[0] == control])
        ctrldist.sort()
        dists = {k: v for k, v in dists.items() if k[0] != control}
        dists[control] = ctrldist
    return dists


def distance_ecdf(res):
    ecdfs = {}
    for k, v in res.items():
        ecdfs[k] = {}
        ecdfs[k]["x"] = v
        ecdfs[k]["y"] = ECDF(v)
    return ecdfs


def distance_wasserstein_1d(
    distr, control="control_transfection_reagent_only", xlim=50
):
    ctrl = distr[control]
    wass = {t: wasserstein_distance(ctrl, distr[t]) for t in distr}
    wass_df = pd.DataFrame.from_dict(wass, orient="index", columns=["wasserstein"])

    trimmed_dists = {t: i[np.where(i < xlim)] for t, i in distr.items()}
    ctrl_trimmed = trimmed_dists[control]
    trimmed_wass = {
        t: wasserstein_distance(ctrl_trimmed, trimmed_dists[t]) for t in trimmed_dists
    }
    trimmed_wass_df = pd.DataFrame.from_dict(
        trimmed_wass, orient="index", columns=["wasserstein_trimmed"]
    )

    return pd.concat([wass_df, trimmed_wass_df], axis=1)


def _interpolate_ecdfs(ecdfs, resolution=1000, xlim=50):
    bins = np.linspace(0, xlim, resolution, endpoint=True)
    interp = {}
    for k, v in ecdfs.items():
        interp[k] = v["y"](bins)
    interp = pd.DataFrame.from_dict(interp)
    interp.insert(0, "x", bins)
    return interp


def _compute_ks(ecdfs, control="control_transfection_reagent_only"):
    ctrl = ecdfs[control]["x"]
    tests = {t: kstest(ctrl, ecdfs[t]["x"]) for t in ecdfs}
    results = {t: i.statistic * i.statistic_sign for t, i in tests.items()}
    pvals = {t: i.pvalue for t, i in tests.items()}
    return results, pvals


def _compute_area(ecdfs, control="control_transfection_reagent_only", resolution=1000):
    interp = _interpolate_ecdfs(ecdfs, resolution=resolution)
    ecdf_no_x = interp.drop("x", axis=1)
    ref = interp[control]
    ecdf_norm = ecdf_no_x.sub(ref, axis=0).abs()
    area = np.trapz(ecdf_norm, x=interp["x"], axis=0) / interp["x"].max()
    area = {t: i for t, i in zip(ecdf_norm.columns, area)}
    return area


def compute_ecdf_stats(
    ecdfs, control="control_transfection_reagent_only", resolution=1000, trim_shift=0.1
):
    ks, kspvals = _compute_ks(ecdfs, control=control)
    area = _compute_area(ecdfs, control=control, resolution=resolution)
    # shift = _compute_shift(ecdfs, control=control, resolution=resolution, trim=trim_shift)
    out = [pd.Series(i) for i in [area, ks, kspvals]]
    out = pd.concat(out, axis=1)
    out.columns = ["area", "ks", "kspval"]
    return out


def read_adata(file, backed=False):
    """Read in single-cell data and rename columns to match metadata"""
    adata = ad.read_h5ad(file, backed="r") if backed else ad.read_h5ad(file)
    if "Experiment" in adata.obs.columns:
        adata.obs.rename(columns={"Experiment": "PlateLayout"})
    return adata


def process_metadata(adata, metadata_regex):
    """Massage metadata to adhere to a common standard"""
    obs = adata.obs

    st = obs["SampleName"].str

    # Expand metadata with regex, keeping original samplename column
    meta = st.extract(metadata_regex, expand=True).drop_duplicates()
    meta.insert(0, "SampleName", obs["SampleName"])

    if "Well" in meta.columns:
        meta["Well"] = meta["Well"].astype(str).astype("category")
    if "Site" in meta.columns:
        meta["Site"] = meta["Site"].astype(int)

    adata.obs = pd.merge(adata.obs, meta, how="left")
    return adata


def add_platemap(adata, platemap):
    """Join platemap to single-cell data"""
    new = adata.obs.merge(platemap, on=["PlateLayout", "Well"], how="left")
    new.index = adata.obs.index
    adata.obs = new
    return adata


def sort_adata(adata, metadata_regex):
    """Sort AnnData by metadata columns"""
    # Get metadata columns from regex
    metadata_regex = re.compile(metadata_regex)
    sort_cols = list(metadata_regex.groupindex.keys())
    # Check that metadata columns are actually in adata
    sort_cols = [col for col in adata.obs.columns if col in sort_cols]
    # Use natural sort to sort anndata
    idx = adata.obs.sort_values(sort_cols, key=natsort_keygen()).index
    return adata[idx].copy()  # avoid view due to slow IO


def filter_adata_by_qc(scadata, qcadata):
    return (
        scadata[scadata.obs["PassQC"] == "True"].copy(),
        qcadata[qcadata.obs["PassQC"] == "True"].copy(),
    )


def read_qc(qc_file, metadata_regex):
    """miRNA-specific QC parser"""
    qc = ad.read_h5ad(qc_file)
    qc = process_metadata(qc, metadata_regex=metadata_regex)
    qc = sort_adata(qc, metadata_regex=metadata_regex)
    return qc


def kNN_dists(adata, pcs=3, neighbors=10, scale=True):
    """
    Compute maximum kNN distance (i.e. radius of smallest enclosing circle of kNNs)

    Parameters
    ----------
    adata : AnnData
        image-level data
    pcs : int, optional
        Number of PCs, by default 3
    neighbors : int, optional
        Number of image neigbors in PC, by default 10
    scale : bool, optional
        Scale by dimensionality to remove higher-dimensional effect of increasing distances, by default True

    Returns
    -------
    np.array
        kNN distances
    """
    from sklearn.neighbors import NearestNeighbors

    a = _get_pca(adata)[:, :pcs]
    nbrs = NearestNeighbors(n_neighbors=neighbors + 1).fit(a)
    d, _ = nbrs.kneighbors(a)
    dss = d[:, 1:]
    dist = dss[:, neighbors - 1]
    return dist / np.sqrt(pcs) if scale else dist


def unsupervised_imageQC(qcadata, pcs=3, neighbors=10, scale=True):
    """
    Compute maximum kNN distance (i.e. radius of smallest enclosing circle of kNNs).
    This function will perform center-scaling and PCA transform, before computing distances.
    It also saves the image-level PCA in obsm["X_pca"].

    Parameters
    ----------
    adata : AnnData
        image-level data
    pcs : int, optional
        Number of PCs, by default 3
    neighbors : int, optional
        Number of image neigbors in PC, by default 10
    scale : bool, optional
        Scale by dimensionality to remove higher-dimensional effect of increasing distances, by default True

    Returns
    -------
    AnnData
        Image-level data with added ImageQCDistance in obs and PCA in obsm
    """
    qcadatac = qcadata.copy()
    embed(qcadatac)
    dists = kNN_dists(qcadatac, pcs, neighbors, scale)
    qcadata.obs["ImageQCDistance"] = dists
    qcadata.obsm["X_pca"] = qcadatac.obsm["X_pca"]
    qcadata.uns["pca"] = qcadatac.uns["pca"]
    qcadata.varm["PCs"] = qcadatac.varm["PCs"]
    return qcadata


def qc(scadata, qcadata, filter=True, threshold=0.05, **kwargs):
    """
    Perform QC of datasets using unsupervised, kNN-based distance filtering

    Parameters
    ----------
    scadata : AnnData
        Single-cell data
    qcadata : AnnData
        Image-level data
    filter : bool, optional
        Whether to return filtered or unfiltered single-cell data, by default True
    threshold : float, optional
        Threshold for removal, by default 0.05

    Returns
    -------
    AnnData
        Single-cell AnnData with QC metrics in obs
    """
    qcadata = unsupervised_imageQC(qcadata, **kwargs)
    qcadata.obs["PassQC"] = qcadata.obs["ImageQCDistance"] < threshold
    qcadata.obs["PassQC"] = qcadata.obs["PassQC"].astype(str).astype("category")
    new = pd.merge(scadata.obs, qcadata.obs, how="left")
    new.index = scadata.obs.index
    scadata.obs = new
    if not filter:
        return scadata, qcadata
    return filter_adata_by_qc(scadata, qcadata)


def count_cells_per_img(
    adata, batch_key="PlateID", well_key="Well", treatment_key="Treatment"
):
    """Helper to count (rounded) number of cells per image"""
    obs = adata.obs
    adata.obs = (
        obs.groupby([batch_key, well_key, treatment_key], observed=True)
        .apply(
            lambda x: np.rint(len(x) / len(x["Site"].unique())), include_groups=False
        )
        .astype(int)
        .rename("CellsPerImage")
        .to_frame()
        .reset_index()
        .merge(obs, how="right")
        .set_index(obs.index)
    )
    return adata


def batch_correct(
    adata,
    copy=False,
    batch_key="PlateID",
    treatment_key="Treatment",
    neg_control="control_transfection_reagent_only",
):
    assert neg_control in adata.obs[treatment_key].unique()
    if copy:
        adata = adata.copy()

    sm.pp.remove_batch_effects(
        adata,
        bio_key=None,
        batch_key=batch_key,
        treatment_key=treatment_key,
        control=neg_control,
    )

    adata.uns["batch_effects"].columns = adata.uns["batch_effects"].columns.astype(str)
    return adata


def _plate_iterator(
    adata,
    keep_all_background=False,
    batch_key="PlateLayout",
    treatment_key="Treatment",
    neg_control="control_transfection_reagent_only",
    copy=False,
):
    obs = adata.obs
    plates = sorted(obs[batch_key].unique())
    for p in plates:
        foreground = obs[obs[batch_key] == p].index
        combined = foreground
        if keep_all_background:
            background = obs[obs[treatment_key] == neg_control].index
            combined = np.concatenate([background, foreground])
            combined = np.unique(combined)
        if copy:
            yield adata[combined].copy(), p
        else:
            yield adata[combined], p


def format_prefix(prefix=""):
    return prefix if prefix == "" else f"_{prefix}_"


def preprocess(
    adata_file,
    metadata_regex,
    platemap_file=None,
    do_batch_correct=True,
    batch_key="PlateID",
    replicate_key="Replicate",
    treatment_key="Treatment",
    neg_control="control_transfection_reagent_only",
):
    pmap = pd.read_csv(platemap_file)
    adata = read_adata(adata_file, backed=False)
    adata = process_metadata(adata, metadata_regex=metadata_regex)
    if platemap_file:
        adata = add_platemap(adata, pmap)
    adata = sort_adata(adata, metadata_regex=metadata_regex)
    sm.pp.drop_na(adata)

    if not do_batch_correct:
        return [adata]

    if replicate_key not in adata.obs.columns:
        raise ValueError(f"Replicate key {replicate_key} not found in metadata")

    # batch correction over all plates
    if (not replicate_key) or len(adata.obs[replicate_key].unique()) == 1:
        adata = batch_correct(
            adata,
            batch_key=batch_key,
            treatment_key=treatment_key,
            neg_control=neg_control,
        )
        return [adata]

    # splitting batch correction over replicates
    repl_adata_list = []
    for repl in adata.obs[replicate_key].unique():
        repl_adata = adata[adata.obs[replicate_key] == repl].copy()
        repl_adata = batch_correct(
            repl_adata,
            batch_key=batch_key,
            treatment_key=treatment_key,
            neg_control=neg_control,
        )
        repl_adata_list.append(repl_adata)
    return repl_adata_list


def qc_adata(
    adata,
    qc_file,
    sample_name,
    metadata_regex,
    out=None,
    prefix="",
    min_cell_count=10,
    **kwargs,
):
    """Perform QC using image-level data"""
    qcadata = read_qc(qc_file, metadata_regex=metadata_regex)
    filtadata, qcadata = qc(adata, qcadata, filter=False, **kwargs)

    if out:
        qcadata.write(f"{out}/{sample_name}_prefilter_qc.h5ad")

    # Perform filtering
    filtadata = filtadata[filtadata.obs["PassQC"] == "True"].copy()

    # Filter for cell counts
    filtadata = count_cells_per_img(filtadata)
    filtadata = filtadata[filtadata.obs["CellsPerImage"] > min_cell_count]

    if out:
        filtadata.write(f"{out}/{sample_name}{prefix}_features_qc.h5ad")
        qcadata.write(f"{out}/{sample_name}_postfilter_qc.h5ad")
    return filtadata


def kruskal_test(adata, batch_feature="PlateID", by=None):
    batch_X = adata.obs[batch_feature].astype(str).astype("category").values
    test_results = {}

    if by is not None:
        iterator = _plate_iterator(adata, batch_key=by, copy=False)
    else:
        iterator = [(adata, "all")]

    for adata_ss, group_id in iterator:
        for selected_feature in tqdm(adata_ss.var.index):
            feature_X = adata_ss[:, selected_feature].X[:, 0]
            batch_indices_d = (
                pd.DataFrame({"batch": batch_X, "feature": feature_X})
                .groupby("batch", observed=True)
                .indices
            )
            feature_split_by_batch = [
                *[feature_X[batch_indices_d[batch]] for batch in batch_indices_d]
            ]
            try:
                res = kruskal(*feature_split_by_batch)
            except ValueError:
                res = SimpleNamespace(statistic=np.nan, pvalue=np.nan)
            test_results[(group_id, selected_feature)] = res

    kruskal_df = pd.DataFrame(
        [
            (
                group_id,
                feature,
                test_results[(group_id, feature)].statistic,
                test_results[(group_id, feature)].pvalue,
            )
            for group_id, feature in test_results
        ],
        columns=["plate", "feature", "statistic", "pvalue"],
    )
    kruskal_df.metadata = SimpleNamespace(by=by, batch_feature=batch_feature)
    if "kruskal_test" not in adata.uns:
        adata.uns["kruskal_test"] = {}
    adata.uns["kruskal_test"][batch_feature] = kruskal_df
    return adata


def kruskal_filter(
    adata, annotate_only=False, sigma=1, batch_feature="PlateID", sigma_function="mad"
):
    def threshold_statistic(adata, batch_feature):
        df = adata.uns["kruskal_test"][batch_feature]
        x = df["statistic"].values
        med = np.median(x)
        if sigma_function == "mad":
            std = median_abs_deviation(x)
        else:
            std = np.std(x)
        thresh = med + sigma * std

        sn = adata.uns["kruskal_test"][batch_feature].metadata
        new = SimpleNamespace(threshold=thresh, median=med, std=std)

        adata.uns["kruskal_test"][batch_feature].metadata = SimpleNamespace(
            **{**sn.__dict__, **new.__dict__}
        )
        return adata

    def filter_threshold_statistic(adata, batch_feature):
        df = adata.uns["kruskal_test"][batch_feature]
        thresh = df.metadata.threshold
        return df.query(f"statistic < {thresh}")["feature"]

    adata = threshold_statistic(adata, batch_feature)

    if annotate_only:
        return adata

    feat_keep = filter_threshold_statistic(adata, batch_feature)
    return adata[:, feat_keep]


def subsample_by(adata, by=["PlateID", "Well"], n_obs=30):
    idx = (
        adata.obs.reset_index()
        .groupby(by, observed=True)
        .apply(lambda x: x.sample(min(n_obs, x.shape[0]), random_state=2024))
        .loc[:, "index"]
        .sort_values()
        .to_numpy()
    )
    return adata[idx, :]


def run_kruskal_filter(
    adata, sample_name, covariates=["PlateID", "Col"], sigma=1, out=None, prefix=""
):
    adata_ss = subsample_by(adata, n_obs=5)
    keep = []
    for c in covariates:
        kruskal_test(adata_ss, batch_feature=c)
        filt = kruskal_filter(adata_ss, batch_feature=c, sigma=sigma).var.index
        keep.extend(filt)
    # keep only features that were not associated with any covariate
    features, fcounts = np.unique(keep, return_counts=True)
    keep = features[fcounts == len(covariates)]
    filtadata = adata[:, keep].copy()
    filtadata.write(f"{out}/{sample_name}{prefix}_features_qc_featFilt.h5ad")
    return filtadata


def unsupervised_ecdf_distances(
    adata,
    out,
    sample_name=None,
    prefix="",
    group_key=None,
    keep_all_background=False,
    batch_key="PlateID",
    treatment_key="Treatment",
    neg_control="control_transfection_reagent_only",
):
    outl = []

    if isinstance(group_key, str):
        group_key = [group_key]

    if keep_all_background:
        # PC embedding over all plates, to make DMSO midoid comparable
        embed(adata)

    for adata_ss, p in _plate_iterator(
        adata,
        copy=not keep_all_background,
        keep_all_background=keep_all_background,
        batch_key=batch_key,
        treatment_key=treatment_key,
        neg_control=neg_control,
    ):
        print(f"Processing {p}")
        if not keep_all_background:
            embed(adata_ss)

        distr = distance_distributions(
            adata_ss, key=treatment_key, control=neg_control, group_key=group_key
        )
        ecdfs = distance_ecdf(distr)
        stats = compute_ecdf_stats(ecdfs, control=neg_control, resolution=1000)
        stat_wasser = distance_wasserstein_1d(distr, control=neg_control)
        stats = pd.merge(
            stats, stat_wasser, how="left", left_index=True, right_index=True
        )

        if group_key is not None:
            ctrl = stats.loc[neg_control].to_frame().T
            ctrl[group_key] = np.nan
            treat = stats[stats.index != neg_control]
            feat_cols = treat.columns
            treat.index = pd.MultiIndex.from_tuples(treat.index)
            treat.index.names = [treatment_key, *group_key]
            treat.reset_index(inplace=True)
            treat.set_index(treatment_key, inplace=True)
            treat = pd.concat([ctrl, treat])
            treat = treat[[*group_key, *feat_cols]]
            stats = treat
        if sample_name is not None:
            stats.insert(0, "sample_name", sample_name)
        if prefix != "":
            stats.insert(1, "Replicate", prefix)
            stats["Replicate"] = stats["Replicate"].str.replace("_", "")
        stats.insert(2, "PlateLayout", p)
        outl.append(stats)

    stats.sort_values("wasserstein", ascending=False, inplace=True)
    stats = pd.concat(outl)
    stats.reset_index(inplace=True, names=treatment_key)
    stats.to_csv(f"{out}/{sample_name}{prefix}_ecdf_stats.csv", index=False)
    return stats
