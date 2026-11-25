import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from georeview.model import run_model


def _businesses():
    rows = []
    for i in range(48):
        group = i % 3
        n = (5, 12, 25)[group] + i % 4
        high = (1, 3, 5)[group]
        ratings = np.array([high] * (n - 2) + [3, min(5, high + i % 3)])
        rows.append({
            "business_id": f"v-{i:03}", "review_count": n, "mean_rating": ratings.mean(),
            "rating_variance": ratings.var(), "positive_ratio": (ratings >= 4).mean(),
            "negative_ratio": (ratings <= 2).mean(), "star_3_ratio": (ratings == 3).mean(),
            "reviews_per_month": np.nan if i % 10 == 0 else n / 5,
            "sentiment_mean": 0.25, "sentiment_coverage": 0.05, "city": "Example city"
        })
    return pd.DataFrame(rows)


class ModelTests(unittest.TestCase):
    def test_candidates_metrics_and_pca(self):
        rows = _businesses()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            summary = run_model(rows, out, k_values=(2, 3, 4))
            self.assertEqual(json.loads((out / "model_summary.json").read_text(encoding="utf-8")), summary)
            assignments = pd.read_csv(out / "assignments.csv")
            metrics = pd.read_csv(out / "metrics.csv")
            profiles = pd.read_csv(out / "profiles.csv")
            agreements = pd.read_csv(out / "distance_agreement.csv")
            winner = metrics.loc[metrics.selected].iloc[0]
            self.assertEqual(winner.config_id, summary["selected_config_id"])
            self.assertEqual(len(metrics), 18)
            self.assertFalse(((metrics.linkage == "ward") & (metrics.distance == "manhattan")).any())
            self.assertTrue(metrics.loc[metrics.algorithm == "Agglomerative", ["wcss", "seed_ari"]].isna().all().all())
            self.assertTrue(metrics.sample_ari.between(-1, 1).all())
            self.assertTrue((metrics.sample_ari_trials == 3).all())
            self.assertEqual(len(agreements), 6)
            self.assertTrue(agreements.ari.between(-1, 1).all())
            self.assertEqual(profiles.business_count.sum(), len(rows))
            self.assertTrue(np.isfinite(assignments[["pc1", "pc2"]]).all().all())
            self.assertEqual(set(assignments.assignment_method), {"fit"})
            self.assertEqual(summary["elbow_curve"][0]["k"], 1)
            self.assertIn("sentiment_mean", summary["excluded_features"])
            # Reconstruct the winning silhouette from exported transformation parameters.
            features = rows.assign(log_review_count=np.log1p(rows.review_count),
                                   log_reviews_per_month=np.log1p(rows.reviews_per_month))[summary["feature_names"]]
            settings = summary["preprocessing"]
            x = (features.fillna(settings["medians"]) - pd.Series(settings["means"])) / pd.Series(settings["scales"])
            self.assertAlmostEqual(silhouette_score(x, assignments.cluster_id, metric=winner.distance), winner.silhouette)

    def test_sampling_is_reproducible_when_input_is_reordered(self):
        rows = _businesses()
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            run_model(rows, out / "first", k_values=(2, 3), max_fit=20)
            run_model(rows.sample(frac=1, random_state=123), out / "again", k_values=(2, 3), max_fit=20)
            first = pd.read_csv(out / "first/assignments.csv")
            second = pd.read_csv(out / "again/assignments.csv")
            pd.testing.assert_frame_equal(first, second)
            self.assertEqual(first.assignment_method.eq("fit").sum(), 20)
            self.assertEqual(len(first), len(rows))
            fit = first.assignment_method == "fit"
            self.assertLessEqual(set(first.loc[~fit, "cluster_id"]), set(first.loc[fit, "cluster_id"]))

    def test_no_evidence_to_segment(self):
        rows = pd.concat([_businesses().iloc[[0]]] * 5, ignore_index=True)
        rows["business_id"] = ["a", "b", "c", "d", "e"]
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "no varying"):
                run_model(rows, folder)
            self.assertFalse((Path(folder) / "model_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
