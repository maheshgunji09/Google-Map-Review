"""Produce a readable analysis alongside the exact measurements for each run."""

import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def category_mix(rows, folder):
    """Sum sparse one-hot service indicators for each final segment."""
    codes, categories = pd.factorize(rows.category.fillna("Unknown"), sort=True)
    groups, ids = pd.factorize(rows.cluster_id, sort=True)
    positions = np.arange(len(rows))
    ones = np.ones(len(rows), dtype=np.int64)
    services = csr_matrix((ones, (positions, codes)), shape=(len(rows), len(categories)))
    clusters = csr_matrix((ones, (groups, positions)), shape=(len(ids), len(rows)))
    totals = (clusters @ services).tocoo()
    result = pd.DataFrame({"cluster_id": ids.take(totals.row),
                           "category": categories.take(totals.col),
                           "businesses": totals.data})
    result["within_segment_share"] = result.businesses / result.groupby("cluster_id").businesses.transform("sum")
    result.sort_values(["cluster_id", "businesses", "category"], ascending=[True, False, True]).to_csv(
        folder / "category_mix.csv", index=False)


def readable(value):
    return str(value).replace("_", " ").replace("|", " / ").replace("\n", " ")


def number(value):
    return f"{value:.3f}" if pd.notna(value) else "unavailable"


def write_report(folder, manifest):
    rows = pd.read_csv(folder / "assignments.csv", dtype={"business_id": str})
    metrics = pd.read_csv(folder / "metrics.csv")
    profiles = pd.read_csv(folder / "profiles.csv")
    contrasts = pd.read_csv(folder / "feature_contrasts.csv")
    agreement = pd.read_csv(folder / "distance_agreement.csv")
    summary = json.loads((folder / "model_summary.json").read_text(encoding="utf-8"))
    category_mix(rows, folder)
    chosen = metrics.loc[metrics.selected.eq(True)].iloc[0]
    date_start = rows.first_review_date.dropna().min()
    date_end = rows.last_review_date.dropna().max()
    dates = (f" Dated reviews span {date_start} through {date_end}."
             if pd.notna(date_start) and pd.notna(date_end) else "")
    if manifest["dataset_kind"] == "synthetic":
        scope = ("**Synthetic demonstration.** Every venue and rating in this run is fictional. "
                 "The results show how the methods behave on the generated inputs. "
                 "Research questions about actual Google Maps businesses in India remain open.")
    else:
        scope = ("**Historical research run.** These results describe eligible businesses in the supplied "
                 "2018 archive. Coverage depends on the source and the location evidence retained by the filter.")
    lines = ["# GeoReview India analysis", "", scope, "", "## Coverage and segmentation", "",
             f"The validated sample contains **{len(rows):,} businesses** with "
             f"**{int(rows.review_count.sum()):,} reviews**. "
             f"The median business has {rows.review_count.median():g} reviews. "
             f"{int((rows.review_count == 1).sum()):,} businesses have a single rating.{dates}", "",
             "### RQ1 · Which experience patterns emerged?", "",
             "| Segment | Businesses | Mean venue rating | Median reviews per venue |",
             "| --- | ---: | ---: | ---: |"]
    for p in profiles.itertuples():
        lines.append(f"| {readable(p.segment)} | {p.business_count:,} | {p.mean_rating:.2f} | {p.median_review_count:g} |")
    lines.extend(["", "Profile means give each business equal weight. Cluster names describe observed evidence.",
                  "", "### RQ2 · Which measurements distinguish the segments?", ""])
    for f in contrasts.dropna(subset=["eta_squared"]).nlargest(3, "eta_squared").itertuples():
        role = "model input" if f.used_in_model else "descriptive feature"
        lines.append(f"- {readable(f.feature)} ({role}) has a between-segment variance share of {f.eta_squared:.1%}.")
    lines += ["", "These shares describe assignments produced by this run. Features derived from the same "
              "star counts are correlated, and these values do not establish causality.",
              "", "### RQ3 · How do the clustering methods compare?", "",
              "Within each method family, the following eligible candidate had the lowest mean rank.",
              "", "| Configuration | Silhouette | Davies–Bouldin | Resampling ARI |",
              "| --- | ---: | ---: | ---: |"]
    scored = metrics.dropna(subset=["selection_score"])
    for algorithm in metrics.algorithm.unique():
        group = scored.loc[scored.algorithm == algorithm]
        if group.empty:
            lines.append(f"| {algorithm} (no eligible configuration) | — | — | — |")
            continue
        best = group.sort_values(["selection_score", "k", "config_id"]).iloc[0]
        lines.append(f"| {best.config_id} | {number(best.silhouette)} | {number(best.davies_bouldin)} | "
                     f"{number(best.sample_ari)} |")
    lines += ["", "Silhouette uses each candidate’s distance. Davies–Bouldin uses Euclidean centroids "
              "for a common compactness measure. "
              + (f"The K-Means WCSS curve suggests an elbow at k = {summary['elbow_k']}."
                 if summary["elbow_k"] else "The tested K-Means WCSS curve has no identifiable elbow."),
              "", "### RQ4 · How much does changing distance alter membership?", ""]
    paired = agreement.dropna(subset=["ari"])
    if len(paired):
        lowest = paired.nsmallest(1, "ari").iloc[0]
        lines.append(f"Across {len(paired)} matched linkage and cluster-count pairs, Euclidean and Manhattan "
                     f"partitions had ARIs from {paired.ari.min():.3f} to {paired.ari.max():.3f}. "
                     f"The lowest agreement occurred with {lowest.linkage} linkage at k = {lowest.k}. "
                     "An ARI of 1 indicates identical partitions up to label permutation.")
    else:
        lines.append("No valid matched pair was available for the requested cluster counts.")
    lines += ["", "### RQ5 · Which configuration was selected?", "",
              f"**{chosen.config_id}** obtained a silhouette of {number(chosen.silhouette)}, "
              f"a Davies–Bouldin index of {number(chosen.davies_bouldin)} and a resampling ARI of "
              f"{number(chosen.sample_ari)}. The smallest fitted cluster contains {int(chosen.min_cluster_size)} businesses. "
              f"The comparison used {summary['n_fit']:,} fitted businesses. The first two PCA axes capture "
              f"{sum(summary['pca_explained_variance_ratio']):.1%} of the fit cohort’s standardized variation.",
              "", "### RQ6 · How can a user explore local services?", "",
              "Select a city and service in the dashboard, then choose the review pattern and minimum evidence "
              "that fit the decision. For a business with many high ratings, check the share of low ratings "
              "and the review count alongside the mean. Sparse feedback provides little evidence about consistency.",
              "", "## Limits of interpretation", "",
              "- Historical reviews cannot establish current quality, availability, or opening hours.",
              "- Archive coverage and geographic validation can favor well-documented urban venues. "
              "Business counts do not measure the prevalence of a pattern across India.",
              "- Self-selection, uneven star use, duplicate content, coordinated activity and missing dates "
              "can alter apparent satisfaction and review frequency. Duplicate removal cannot certify authenticity.",
              "- Text coverage varies across languages and scripts. Missing sentiment is unknown. "
              "Review-level exports omit text and reviewer identity.",
              "- Internal separation and resampling stability describe this sample. They do not validate "
              "service quality or establish which recommendation a customer will prefer.", "",
              "The [geographic and cleaning audit](audit.json), [model settings](model_summary.json), "
              "[full configuration grid](metrics.csv), [feature contrasts](feature_contrasts.csv), "
              "[distance comparisons](distance_agreement.csv) and [service mix](category_mix.csv) "
              "contain the measurements behind this analysis.", ""]
    (folder / "analysis.md").write_text("\n".join(lines), encoding="utf-8")
