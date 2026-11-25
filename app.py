"""Explore completed GeoReview India runs with Streamlit."""

import json
import math
import os
from html import escape
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from georeview.glass import COMPONENT, overview


st.set_page_config(page_title="GeoReview India", page_icon=":material/travel_explore:", layout="wide")
glass_overview = st.components.v2.component(**COMPONENT)

OUTPUTS = Path(__file__).resolve().parent / "outputs"
STAR_COLUMNS = [f"star_{star}_ratio" for star in range(1, 6)]
FILES = ("manifest.json", "model_summary.json", "assignments.csv", "profiles.csv",
         "metrics.csv", "feature_contrasts.csv", "distance_agreement.csv")
STYLES = """
<style>
.stMainBlockContainer {max-width:1360px;padding-top:2.4rem;}
[data-testid="stAppViewContainer"] {background:radial-gradient(ellipse at 5% 0%,#e5f3f1 0,transparent 50%),radial-gradient(ellipse at 95% 15%,#e9e8f7 0,transparent 50%),#f5f8fb;}
[data-testid="stHeader"] {background:rgba(245,248,251,.85);backdrop-filter:blur(16px);}
[data-baseweb="tab-list"] {padding:7px;gap:6px;border-radius:20px;background:#ffffffa6;border:1px solid white;box-shadow:0 5px 22px #2e49690b;}
[data-baseweb="tab"] {border-radius:14px;padding:10px 18px;height:auto;}
[data-baseweb="tab"][aria-selected="true"] {background:white;box-shadow:0 2px 8px #2e496918;color:#1b607c;}
[data-baseweb="tab-highlight"],[data-baseweb="tab-border"] {display:none;}
[data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] > div {background:#ffffffb8;border-color:#ffffff!important;border-radius:20px;box-shadow:0 8px 28px #2844680b;backdrop-filter:blur(16px);}
[data-testid="stExpander"] {background:#ffffffb8;border-radius:16px;}
[data-testid="stDataFrame"] {border-radius:16px;overflow:hidden;box-shadow:0 6px 25px #2844680a;}
[data-baseweb="select"] > div,[data-testid="stNumberInputContainer"] {background:#ffffffc7;box-shadow:inset 0 1px 0 white;}
button:focus-visible {outline:3px solid #1b607c!important;outline-offset:3px;}
.gr-name {font:600 1.7rem/1.2 Inter,system-ui,sans-serif;letter-spacing:-.6px;color:#172b3d;margin:0;}
.gr-meta {color:#486075;margin:.45rem 0;}
@media(max-width:700px) {.stMainBlockContainer{padding:2.8rem 1rem 1rem}[data-baseweb="tab"]{padding:10px 12px;white-space:nowrap}}
@media(prefers-reduced-transparency:reduce),(prefers-contrast:more) {[data-testid="stAppViewContainer"]{background:#f5f8fb}[data-baseweb="tab-list"],[data-testid="stMetric"],[data-testid="stExpander"]{background:white;backdrop-filter:none}}
</style>
"""


def run_directory():
    override = os.environ.get("GEOREVIEW_RUN")
    if override:
        return Path(override).expanduser()
    research = OUTPUTS / "research"
    return research if all((research / f).is_file() for f in FILES) else OUTPUTS / "demo"


@st.cache_data(max_entries=4, show_spinner=False)
def load_run(folder, signature):
    """The file signature invalidates cached data when a run is replaced."""
    path = Path(folder)
    manifest, model = [json.loads((path / name).read_text(encoding="utf-8")) for name in FILES[:2]]
    if manifest.get("dataset_kind") not in {"synthetic", "research"}:
        raise ValueError("The manifest must identify this run as synthetic or research.")
    assignments = pd.read_csv(path / "assignments.csv", dtype={"business_id": str})
    profiles, metrics, features, distances = [pd.read_csv(path / name) for name in FILES[3:]]
    return manifest, model, assignments, profiles, metrics, features, distances


def options_filter(column, label, all_label):
    return st.selectbox(label, [None, *sorted(column.dropna().unique())],
                        format_func=lambda item: all_label if item is None else str(item))


def sorted_matches(frame, preference):
    work = frame.copy()
    if preference == "Supported positive feedback":
        # Wilson's lower bound for the fraction of four- and five-star ratings.
        p, n, z = work.positive_ratio, work.review_count, 1.96
        work["support"] = (p + z**2 / (2*n) - z*np.sqrt((p*(1-p) + z**2/(4*n))/n)) / (1 + z**2/n)
        keys, directions = ["support", "review_count"], [False, False]
    elif preference == "Least rating variation":
        # One observation cannot support a comparison of rating consistency.
        work = work[work.review_count.ge(2) & work.rating_variance.notna()]
        keys, directions = ["rating_variance", "review_count"], [True, False]
    else:
        keys, directions = ["review_count", "positive_ratio"], [False, False]
    return work.sort_values(keys + ["name", "business_id"], ascending=directions + [True, True])


def business_detail(row, profiles):
    with st.container(border=True):
        st.caption("Business detail")
        st.html(f'<p class="gr-name">{escape(str(row["name"]))}</p><p class="gr-meta">'
                f'{escape(str(row["city"]))} · {escape(str(row["category"]))}</p>')
        profile = profiles.loc[profiles.cluster_id.eq(row.cluster_id)].iloc[0]
        st.write(f"**{profile.segment}**")
        st.caption(profile.description)
        ratings = pd.DataFrame({"Stars": ["1 ★", "2 ★", "3 ★", "4 ★", "5 ★"],
                                "Share": [row[c] for c in STAR_COLUMNS]})
        bars = alt.Chart(ratings).mark_bar(color="#27836D", cornerRadiusEnd=3, size=13).encode(
            x=alt.X("Share:Q", title=None, scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%", tickCount=3)),
            y=alt.Y("Stars:N", title=None, sort="descending"),
            tooltip=["Stars:N", alt.Tooltip("Share:Q", format=".1%")])
        st.altair_chart(bars.properties(height=124), width="stretch")
        st.caption(f'{int(row.review_count):,} reviews · {row.mean_rating:.2f} mean stars · '
                   f'{row.negative_ratio:.0%} one- or two-star ratings')
        if pd.notna(row.first_review_date) and pd.notna(row.last_review_date):
            dates = f"Observed from {row.first_review_date} to {row.last_review_date}."
            if pd.notna(row.reviews_per_month):
                dates += f" {row.reviews_per_month:.1f} dated reviews per month."
            st.caption(dates)
        if pd.notna(row.get("sentiment_mean")):
            st.caption(f"Mean text sentiment {row.sentiment_mean:+.2f} on a −1 to +1 scale. "
                       f"Text scoring covered {row.sentiment_coverage:.0%} of reviews.")
        if row.assignment_method == "nearest_medoid":
            st.caption("This business was outside the fit sample. Its segment is an approximation based on the nearest cluster medoid.")
        elif row.assignment_method == "nearest_centroid":
            st.caption("This business was outside the fit sample and was assigned to its nearest K-Means centroid.")


def explorer(data, profiles):
    st.header("City service explorer")
    first, second, third = st.columns(3)
    with first:
        city = options_filter(data.city, "City", "All cities")
    local = data if city is None else data[data.city.eq(city)]
    with second:
        category = options_filter(local.category, "Service category", "All categories")
    local = local if category is None else local[local.category.eq(category)]
    with third:
        segment = options_filter(local.segment, "Review pattern", "All patterns")
    if segment is not None:
        local = local[local.segment.eq(segment)]
    left, right = st.columns([2, 1])
    with left:
        preference = st.selectbox("Prioritize", ["Supported positive feedback", "Least rating variation", "Most reviews"])
    with right:
        minimum = st.number_input("Minimum reviews", min_value=1, value=1, step=1)
    matches = sorted_matches(local[local.review_count >= minimum], preference)
    explanations = {
        "Supported positive feedback": "Ordered by the 95% Wilson lower bound for the share of four- and five-star ratings. More reviews give stronger support for the same share.",
        "Least rating variation": "Ordered by the smallest rating variance among businesses with at least two reviews. A small variance means customers gave similar ratings at any star level.",
        "Most reviews": "Ordered by observed review count. A larger sample gives more evidence about the distribution of experiences.",
    }
    st.caption(explanations[preference])
    count = len(matches)
    if not count:
        st.info("No businesses match these filters. Try a lower review minimum or a broader review pattern.")
        return
    table, detail = st.columns([1.8, 1], gap="large")
    with table:
        st.subheader(f"{count:,} matching {'business' if count == 1 else 'businesses'}")
        page_size = 25
        pages = math.ceil(count/page_size)
        page = st.number_input("Results page", min_value=1, max_value=pages, step=1) if pages > 1 else 1
        visible = matches.iloc[(page-1)*page_size:page*page_size]
        st.caption(f"Showing {(page-1)*page_size+1}–{min(page*page_size, count)}. Select a row to inspect its reviews.")
        shown = visible[["name", "city", "category", "segment", "review_count", "mean_rating",
                         "positive_ratio", "rating_variance"]].copy()
        shown["positive_ratio"] *= 100
        selection = st.dataframe(shown, hide_index=True, width="stretch",
            height=min(445, 39 + len(visible)*35), on_select="rerun", selection_mode="single-row",
            column_config={
                "name": "Business", "city": "City", "category": "Category", "segment": "Review pattern",
                "review_count": "Reviews", "mean_rating": st.column_config.NumberColumn("Mean ★", format="%.2f"),
                "positive_ratio": st.column_config.NumberColumn("4–5 ★", format="%.0f%%"),
                "rating_variance": st.column_config.NumberColumn("Variance", format="%.2f"),
            })
        index = selection.selection.rows[0] if selection.selection.rows else 0
    with detail:
        business_detail(visible.iloc[index if index < len(visible) else 0], profiles)


def segments(data, profiles, features, model):
    st.header("Segment profiles")
    st.caption("Each point represents one business. Nearby points have similar values on the two principal components shown.")
    sample = data if len(data) <= 2500 else data.sample(2500, random_state=model["random_seed"])
    ratios = model["pca_explained_variance_ratio"]
    scatter = alt.Chart(sample[["pc1", "pc2", "segment", "name", "city", "review_count"]]).mark_circle(
        size=80, opacity=.76, stroke="white", strokeWidth=.6).encode(
            x=alt.X("pc1:Q", title=f"Principal component 1 ({ratios[0]:.1%})"),
            y=alt.Y("pc2:Q", title=f"Principal component 2 ({ratios[1]:.1%})"),
            color=alt.Color("segment:N", title="Review pattern", legend=alt.Legend(orient="bottom", columns=1)),
            tooltip=[alt.Tooltip("name:N", title="Business"), alt.Tooltip("city:N", title="City"),
                     alt.Tooltip("segment:N", title="Pattern"), alt.Tooltip("review_count:Q", title="Reviews")])
    st.altair_chart(scatter.properties(height=335).interactive(), width="stretch")
    st.caption(f"Showing {len(sample):,} of {len(data):,} venues. These components capture {sum(ratios):.1%} of variation in the model inputs.")
    chosen = st.selectbox("Inspect a segment", profiles.cluster_id.tolist(),
                          format_func=lambda k: profiles.loc[profiles.cluster_id.eq(k), "segment"].iloc[0])
    profile = profiles.loc[profiles.cluster_id.eq(chosen)].iloc[0]
    st.write(profile.description)
    with st.container(horizontal=True):
        st.metric("Businesses", f"{int(profile.business_count):,}", border=True)
        st.metric("Median reviews", f"{profile.median_review_count:,.1f}", border=True)
        st.metric("Mean rating", f"{profile.mean_rating:.2f} ★", border=True)
        st.metric("Mean positive share", f"{profile.mean_positive_ratio:.0%}", border=True)
    with st.expander("Features that distinguish the segments"):
        st.caption("Eta squared measures the share of each feature’s variation associated with differences between these fitted groups. The values describe this run.")
        display = features.sort_values("eta_squared", ascending=False).copy()
        display["feature"] = display.feature.str.replace("_", " ").str.capitalize()
        st.dataframe(display[["feature", "eta_squared", "used_in_model", "observed_businesses"]],
                     hide_index=True, column_config={"feature": "Feature", "used_in_model": "Model input",
                     "observed_businesses": "Businesses with values",
                     "eta_squared": st.column_config.ProgressColumn("Eta squared", min_value=0, max_value=1, format="%.3f")})


def evaluation(metrics, model, distances):
    st.header("Model comparison")
    winner = metrics.loc[metrics.config_id.eq(model["selected_config_id"])].iloc[0]
    rule = "nearest centroid" if winner.algorithm == "K-Means" else f"{winner.linkage} linkage"
    st.write(f"**{winner.algorithm} with {int(winner.k)} clusters** · {rule} · {winner.distance} distance")
    st.caption(f"Configurations were compared on {model['n_fit']:,} of {model['n_businesses']:,} businesses.")
    with st.container(horizontal=True):
        for label, key in [("Silhouette", "silhouette"), ("Davies–Bouldin", "davies_bouldin"), ("Resample ARI", "sample_ari")]:
            value = winner[key]
            st.metric(label, f"{value:.3f}" if pd.notna(value) else "Unavailable", border=True)
    st.subheader("K-Means elbow")
    elbow = pd.DataFrame(model["elbow_curve"])
    if not elbow.empty:
        curve = alt.Chart(elbow).mark_line(point=True, color="#136B59").encode(
            x=alt.X("k:O", title="Number of clusters"), y=alt.Y("wcss:Q", title="Within-cluster sum of squares"),
            tooltip=[alt.Tooltip("k:O", title="Clusters"), alt.Tooltip("wcss:Q", title="WCSS", format=".2f")])
        st.altair_chart(curve.properties(height=200), width="stretch")
    elbow_k = model.get("elbow_k")
    st.caption(f"The estimated bend occurs at k = {elbow_k}." if elbow_k else "The evaluated curve has no estimated elbow.")
    st.subheader("Configuration results")
    st.caption("Higher silhouette values indicate stronger separation. Lower Davies–Bouldin values indicate more compact and separated groups in Euclidean space. ARI values near 1 indicate agreement across repeated fits.")
    columns = ["selected", "algorithm", "linkage", "distance", "k", "silhouette", "davies_bouldin", "seed_ari", "sample_ari", "min_cluster_size", "wcss"]
    labels = ["Selected", "Algorithm", "Linkage / rule", "Distance", "Clusters", "Silhouette", "Davies–Bouldin", "Seed ARI", "Resample ARI", "Smallest cluster", "WCSS"]
    st.dataframe(metrics[columns].rename(columns=dict(zip(columns, labels))), hide_index=True, width="stretch")
    st.caption("Seed ARI and WCSS apply to K-Means. Hierarchical clustering is deterministic on a fixed input.")
    with st.expander("Selection and stability"):
        st.write(model["selection_rule"])
        st.write(model["stability"])
        support = model["min_cluster_support"]
        if model["support_fallback"]:
            st.write(f"No partition met the target of {support} businesses in every cluster. The comparison included all configurations.")
        else:
            st.write(f"The selection required at least {support} businesses in every cluster.")
    with st.expander("How much did changing the distance alter the groups?"):
        st.caption("Each ARI compares the Euclidean and Manhattan assignments with the same linkage and cluster count. A value of 1 means identical group membership.")
        st.dataframe(distances[["linkage", "k", "ari"]].rename(columns={
            "linkage": "Linkage", "k": "Clusters", "ari": "Distance comparison ARI"}), hide_index=True)


def provenance(manifest, data):
    st.header("Dataset and scope")
    st.subheader(manifest["source_title"])
    st.write(manifest["provenance_note"])
    audit_path = directory / "audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        with st.expander("Geographic validation and cleaning"):
            for policy in ("geography_policy", "city_policy", "frequency_policy", "sentiment_policy", "duplicate_policy"):
                st.write(audit.get(policy, ""))
            st.dataframe(pd.DataFrame([
                {"Audit measure": key.replace("_", " "), "Count": value}
                for key, value in audit.get("counts", {}).items()
            ]), hide_index=True)
    with st.expander("Coverage by city"):
        by_city = (data.groupby("city").agg(businesses=("business_id", "size"), reviews=("review_count", "sum"))
                   .sort_values("businesses", ascending=False).reset_index())
        st.dataframe(by_city.rename(columns={"city": "City", "businesses": "Businesses", "reviews": "Reviews"}),
                     hide_index=True)
    st.markdown("""
**Geography and sampling.** Coverage depends on which businesses appear in the archive and which locations can be verified. Differences between cities may reflect unequal coverage.

**Time and ratings.** Historical review distributions can change as staff, ownership or customers change. Self-selection, inflated ratings and coordinated reviews can affect every segment.

**Text and privacy.** Language coverage and text availability limit sentiment estimates. This dashboard displays business-level aggregates. Small review counts provide limited evidence about typical experiences.
""")


st.html(STYLES)
directory = run_directory()
try:
    signature = tuple((directory / name).stat().st_mtime_ns for name in FILES)
    manifest, model, data, profiles, metrics, features, distances = load_run(str(directory), signature)
except (OSError, ValueError, pd.errors.ParserError) as exc:
    st.title("GeoReview India")
    st.error("A complete analysis run is needed to display the atlas.")
    st.caption(f"Run directory: {directory}. {exc}")
    st.code("python -m georeview demo", language="bash")
    st.stop()

overview(data, manifest["dataset_kind"] == "synthetic", glass_overview)
explore_tab, profiles_tab, models_tab, scope_tab = st.tabs(
    ["City service explorer", "Segment profiles", "Model comparison", "Dataset and scope"],
    key="atlas_view", on_change="rerun")
if explore_tab.open:
    with explore_tab:
        explorer(data, profiles)
if profiles_tab.open:
    with profiles_tab:
        segments(data, profiles, features, model)
if models_tab.open:
    with models_tab:
        evaluation(metrics, model, distances)
if scope_tab.open:
    with scope_tab:
        provenance(manifest, data)
