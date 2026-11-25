"""Compare review-pattern segmentations on one reproducible cohort of businesses."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage as build_tree
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from threadpoolctl import threadpool_limits


_DISTANCES = {"euclidean": "euclidean", "manhattan": "cityblock"}
_TREES = (("ward", "euclidean"), ("average", "euclidean"), ("average", "manhattan"),
          ("complete", "euclidean"), ("complete", "manhattan"))
_REQUIRED = ("business_id", "review_count", "mean_rating", "rating_variance",
             "star_3_ratio", "positive_ratio", "negative_ratio", "reviews_per_month")


def _features(rows):
    # Means and variance summarize location and dispersion. Three-star mass adds
    # information about the shape without counting every related share twice.
    values = pd.DataFrame(index=rows.index)
    values["mean_rating"] = rows.mean_rating
    values["rating_variance"] = rows.rating_variance.where(rows.review_count >= 2)
    values["star_3_ratio"] = rows.star_3_ratio
    values["log_review_count"] = np.log1p(rows.review_count)
    values["log_reviews_per_month"] = np.log1p(rows.reviews_per_month.where(rows.reviews_per_month >= 0))
    if {"sentiment_mean", "sentiment_coverage"}.issubset(rows.columns):
        values["sentiment_mean"] = rows.sentiment_mean.where(rows.sentiment_coverage >= 0.5)
    return values.replace([np.inf, -np.inf], np.nan)


def _transform(values, fit_rows):
    fit = values.iloc[fit_rows]
    # A subsample can be entirely missing a feature selected on the main cohort.
    # A neutral constant gives that feature zero influence within this refit.
    medians = fit.median().fillna(0.0)
    filled = fit.fillna(medians)
    centers = filled.mean()
    scales = filled.std(ddof=0).replace(0, 1.0)
    result = (values.fillna(medians) - centers) / scales
    settings = {"medians": medians.to_dict(), "means": centers.to_dict(), "scales": scales.to_dict()}
    return np.ascontiguousarray(result, dtype=float), settings


def _partitions(matrix, ks, seed):
    partitions = {}
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=seed, algorithm="lloyd")
        labels = model.fit_predict(matrix)
        partitions[("K-Means", "centroid", "euclidean", k)] = labels

    for method, metric in _TREES:
        tree = build_tree(matrix, method=method, metric=_DISTANCES[metric])
        cuts = cut_tree(tree, n_clusters=ks)
        for pos, k in enumerate(ks):
            partitions[("Agglomerative", method, metric, k)] = cuts[:, pos]
    return partitions


def _id(key):
    algorithm, method, metric, k = key
    name = "kmeans" if algorithm == "K-Means" else "agglomerative"
    return f"{name}-{method}-{metric}-k{k}"


def _choose(metrics, n):
    # Prefer a useful minimum support. On very small cohorts every partition
    # may have a singleton, so retain a result and report that limitation.
    support = max(2, int(np.ceil(0.01 * n)))
    eligible = metrics.min_cluster_size >= support
    fallback = not eligible.any()
    if fallback:
        eligible = pd.Series(True, index=metrics.index)
    pool = metrics.loc[eligible]
    # Equivalent partitions can differ in the last few floating-point digits
    # when two estimators number their clusters in a different order.
    scores = pd.concat([
        pool.silhouette.round(12).rank(ascending=False),
        pool.davies_bouldin.round(12).rank(),
        pool.sample_ari.round(12).rank(ascending=False, na_option="bottom"),
    ], axis=1).mean(axis=1)
    metrics["selection_score"] = scores
    ordered = metrics.loc[eligible].sort_values(
        ["selection_score", "k", "algorithm", "config_id"],
        ascending=[True, True, False, True], kind="stable")
    winner = ordered.index[0]
    metrics["selected"] = metrics.index == winner
    return str(metrics.at[winner, "config_id"]), support, fallback


def _elbow(curve):
    # Furthest point below the normalized chord joining the two endpoints.
    # Straight lines and curves with too few points have no detectable bend.
    if len(curve) < 3:
        return None
    k = np.array([p["k"] for p in curve])
    w = np.array([p["wcss"] for p in curve])
    if w[0] <= w[-1] or np.any(np.diff(w) > 1e-8):
        return None
    gap = 1 - (k - k[0]) / (k[-1] - k[0]) - (w - w[-1]) / (w[0] - w[-1])
    index = int(np.argmax(gap))
    return int(k[index]) if gap[index] > 1e-8 else None


def _evaluate(values, matrix, fit_index, ks, seed, rng):
    base = matrix[fit_index]
    partitions = _partitions(base, ks, seed)
    seed_scores = {k: [] for k in ks}
    sample_scores = {key: [] for key in partitions}
    for _ in range(3):
        trial_seed = int(rng.integers(0, 2**31))
        for k in ks:
            refit = KMeans(n_clusters=k, n_init=10, random_state=trial_seed).fit_predict(base)
            reference = partitions[("K-Means", "centroid", "euclidean", k)]
            seed_scores[k].append(adjusted_rand_score(reference, refit))

        positions = np.sort(rng.choice(len(base), size=int(0.8 * len(base)), replace=False))
        subset = values.iloc[fit_index[positions]]
        trial, _ = _transform(subset, np.arange(len(subset)))
        unique = len(np.unique(trial, axis=0))
        trial_ks = [k for k in ks if k <= unique and k < len(trial)]
        if trial_ks:
            for key, labels in _partitions(trial, trial_ks, trial_seed).items():
                reference = partitions[key][positions]
                if 1 < len(np.unique(reference)) < len(reference):
                    sample_scores[key].append(adjusted_rand_score(reference, labels))

    pairwise = {metric: squareform(pdist(base, code)) for metric, code in _DISTANCES.items()}
    records = []
    for key, labels in partitions.items():
        algorithm, method, distance, k = key
        groups, counts = np.unique(labels, return_counts=True)
        if len(groups) != k:
            continue
        wcss = np.nan
        if algorithm == "K-Means":
            wcss = sum(((base[labels == g] - base[labels == g].mean(axis=0))**2).sum() for g in groups)
        scores = sample_scores[key]
        records.append({
            "config_id": _id(key), "algorithm": algorithm, "linkage": method, "distance": distance,
            "k": k, "silhouette": silhouette_score(pairwise[distance], labels, metric="precomputed"),
            "davies_bouldin": davies_bouldin_score(base, labels), "wcss": wcss,
            "seed_ari": np.mean(seed_scores[k]) if algorithm == "K-Means" else np.nan,
            "sample_ari": np.mean(scores) if scores else np.nan,
            "sample_ari_trials": len(scores), "min_cluster_size": int(min(counts))
        })
    return pd.DataFrame(records), partitions, pairwise


def _assign(matrix, fit_index, labels, key, distances):
    algorithm, _, metric, _ = key
    base = matrix[fit_index]
    groups = np.unique(labels)
    all_labels = np.empty(len(matrix), dtype=int)
    all_labels[fit_index] = labels
    methods = np.full(len(matrix), "fit", dtype=object)
    outside = np.setdiff1d(np.arange(len(matrix)), fit_index)
    if len(outside):
        representatives = []
        for group in groups:
            members = np.flatnonzero(labels == group)
            if algorithm == "K-Means":
                representatives.append(base[members].mean(axis=0))
            else:
                totals = distances[np.ix_(members, members)].sum(axis=1)
                representatives.append(base[members[np.argmin(totals)]])
        for batch in np.array_split(outside, max(1, int(np.ceil(len(outside) / 10000)))):
            nearest = cdist(matrix[batch], representatives, metric=_DISTANCES[metric]).argmin(axis=1)
            all_labels[batch] = groups[nearest]
        methods[outside] = "nearest_centroid" if algorithm == "K-Means" else "nearest_medoid"
    return all_labels, methods


def _segment_name(profile, high_volume):
    if profile.median_review_count <= 3:
        return "Limited review history"
    if profile.mean_rating >= 4 and profile.mean_positive_ratio >= 0.75 and profile.mean_rating_variance <= 0.8:
        return "Consistently positive reviews"
    if profile.mean_rating <= 3.0 or profile.mean_negative_ratio >= 0.4:
        return "Lower-rated experiences"
    if profile.mean_rating_variance >= 1.1 and profile.median_review_count >= high_volume:
        return "High-volume mixed reviews"
    if profile.mean_rating_variance >= 1.1:
        return "Mixed customer feedback"
    if profile.mean_rating >= 4:
        return "Generally positive reviews"
    return "Moderate ratings"


def _profile(rows):
    group = rows.assign(observed_variance=rows.rating_variance.where(rows.review_count >= 2)).groupby("cluster_id")
    profiles = group.agg(
        business_count=("business_id", "size"), median_review_count=("review_count", "median"),
        mean_rating=("mean_rating", "mean"), mean_rating_variance=("observed_variance", "mean"),
        mean_positive_ratio=("positive_ratio", "mean"), mean_negative_ratio=("negative_ratio", "mean"),
        mean_reviews_per_month=("reviews_per_month", "mean"),
    ).reset_index()
    profiles["segment"] = [_segment_name(p, max(10, rows.review_count.quantile(0.75)))
                           for p in profiles.itertuples()]
    repeated = profiles.segment.duplicated(keep=False)
    profiles.loc[repeated, "segment"] += " (group " + profiles.loc[repeated, "cluster_id"].astype(str) + ")"
    profiles["description"] = [
        f"Businesses average {p.mean_rating:.1f} stars. The median business has "
        f"{p.median_review_count:g} reviews. On average, {p.mean_positive_ratio:.0%} of ratings "
        f"are 4–5 stars and {p.mean_negative_ratio:.0%} are 1–2 stars."
        for p in profiles.itertuples()
    ]
    return profiles


def _contrasts(rows, values, labels):
    reported = values.copy()
    for name in [*(f"star_{i}_ratio" for i in range(1, 6)), "positive_ratio", "negative_ratio"]:
        if name in rows:
            reported[name] = rows[name]
    results = []
    for column in reported:
        data = reported[column].dropna()
        variance = ((data - data.mean()) ** 2).sum()
        if variance <= 0:
            continue
        grouped = data.groupby(labels.loc[data.index])
        between = (grouped.size() * (grouped.mean() - data.mean()) ** 2).sum()
        results.append({"feature": column, "eta_squared": float(np.clip(between / variance, 0, 1)),
                        "used_in_model": column in values, "observed_businesses": len(data)})
    return pd.DataFrame(results).sort_values("eta_squared", ascending=False, kind="stable")


def run_model(businesses, output_dir, *, k_values=(2, 3, 4, 5, 6), random_seed=42, max_fit=2000):
    """Fit, evaluate and export candidate configurations; return the selected model's summary.

    Accepts an ingestion CSV path or its DataFrame. All candidate models and
    metrics use the same fit cohort. Large cohorts are sampled before fitting.
    """
    rows = (businesses.copy() if isinstance(businesses, pd.DataFrame) else
            pd.read_csv(businesses, dtype={"business_id": "string"}))
    missing = sorted(set(_REQUIRED) - set(rows))
    if missing:
        raise ValueError("Missing business feature columns: " + ", ".join(missing))
    if rows.business_id.isna().any() or rows.business_id.astype(str).duplicated().any():
        raise ValueError("Each row must have a unique, nonmissing business_id.")
    rows["business_id"] = rows.business_id.astype(str)
    rows = rows.sort_values("business_id").reset_index(drop=True)
    for name in {*_REQUIRED[1:], "sentiment_mean", "sentiment_coverage"} & set(rows):
        rows[name] = pd.to_numeric(rows[name], errors="raise")
    if not (rows.review_count.ge(1).all() and np.isfinite(rows.review_count).all()
            and rows.mean_rating.between(1, 5).all()):
        raise ValueError("Businesses need a positive review count and a mean rating between 1 and 5.")
    n = len(rows)
    if min(n, max_fit) < 4:
        raise ValueError("At least four businesses in the fitting cohort are needed for a comparison.")
    rng = np.random.default_rng(random_seed)
    fit_index = np.sort(rng.choice(n, size=max_fit, replace=False)) if n > max_fit else np.arange(n)
    values = _features(rows)
    reference = values.iloc[fit_index]
    features = []
    excluded = {}
    for name in values:
        threshold = 0.8 if name == "sentiment_mean" else 0.5 if name == "log_reviews_per_month" else 0
        if reference[name].notna().mean() < threshold:
            excluded[name] = "insufficient coverage in fit cohort"
        elif reference[name].nunique() < 2:
            excluded[name] = "constant or unavailable in fit cohort"
        else:
            features.append(name)
    if not features:
        raise ValueError("The fit cohort has no varying review-pattern features.")
    values = values[features]
    matrix, preprocessing = _transform(values, fit_index)
    base = matrix[fit_index]
    unique = len(np.unique(base, axis=0))
    requested = sorted(set(k_values))
    ks = [k for k in requested if 2 <= k <= unique and k < len(base) and k < int(0.8 * len(base))]
    if not ks:
        raise ValueError("No requested k fits the number of distinct businesses and stability subsample size.")
    with threadpool_limits(limits=1):
        metrics, partitions, distances = _evaluate(values, matrix, fit_index, ks, random_seed, rng)
        selected, support, fallback = _choose(metrics, len(base))
        key = next(key for key in partitions if _id(key) == selected)
        labels, methods = _assign(matrix, fit_index, partitions[key], key, distances[key[2]])
        pca = PCA(n_components=min(2, len(features)), svd_solver="full").fit(base)
        scores = pca.transform(matrix)
    # Number labels by average review rating to make display order reproducible.
    order = rows.assign(group=labels).groupby("group").mean_rating.mean().sort_values(ascending=False, kind="stable")
    display_ids = {group: i for i, group in enumerate(order.index, start=1)}
    rows["cluster_id"] = pd.Series(labels).map(display_ids)
    profiles = _profile(rows)
    rows["segment"] = rows.cluster_id.map(profiles.set_index("cluster_id").segment)
    rows["assignment_method"] = methods
    rows["pc1"] = scores[:, 0]
    rows["pc2"] = scores[:, 1] if scores.shape[1] > 1 else 0.0

    agreement = []
    for k in ks:
        for method in ("average", "complete"):
            euclid = ("Agglomerative", method, "euclidean", k)
            manhattan = ("Agglomerative", method, "manhattan", k)
            agreement.append({"linkage": method, "k": k, "euclidean_config_id": _id(euclid),
                              "manhattan_config_id": _id(manhattan),
                              "ari": adjusted_rand_score(partitions[euclid], partitions[manhattan])})
    curve = [{"k": 1, "wcss": float(((base - base.mean(axis=0))**2).sum())}]
    curve.extend(metrics.loc[metrics.algorithm == "K-Means", ["k", "wcss"]].to_dict("records"))
    explained = pca.explained_variance_ratio_.tolist()
    summary = {
        "selected_config_id": selected, "n_businesses": n, "n_fit": len(base),
        "pca_explained_variance_ratio": (explained + [0.0])[:2], "feature_names": features,
        "excluded_features": excluded, "preprocessing": preprocessing,
        "random_seed": random_seed, "max_fit": max_fit, "requested_k": requested, "tested_k": ks,
        "elbow_k": _elbow(curve), "elbow_curve": curve,
        "min_cluster_support": support, "support_fallback": fallback,
        "selection_rule": ("Smallest mean rank across native-distance silhouette (higher), Euclidean "
                           "Davies–Bouldin (lower) and resampling ARI (higher), among partitions meeting "
                           "minimum support if available. Ties prefer fewer groups, then K-Means."),
        "stability": ("Mean ARI against the reference partition over three additional seeds (K-Means) "
                      "and three random 80% subsets refitted with new medians and scales. Subset ARI "
                      "uses shared businesses with a nontrivial reference partition."),
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows.to_csv(out / "assignments.csv", index=False)
    profiles.to_csv(out / "profiles.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(agreement).to_csv(out / "distance_agreement.csv", index=False)
    _contrasts(rows.iloc[fit_index], values.iloc[fit_index], rows.cluster_id).to_csv(
        out / "feature_contrasts.csv", index=False)
    (out / "model_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
