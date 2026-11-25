"""Exercise the public CLI through ingestion, model selection and the report."""

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from georeview.demo import write_demo


ROOT = Path(__file__).resolve().parents[1]


def records(folder, name):
    with (folder / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class WorkflowTest(unittest.TestCase):
    def test_seeded_run_has_auditable_counts_and_private_exports(self):
        with tempfile.TemporaryDirectory() as work:
            folder = Path(work)
            output = folder / "run"
            command = [sys.executable, "-m", "georeview", "demo", "--output", str(output),
                       "--k", "2", "3", "--max-fit", "64"]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset_kind"], "synthetic")
            inputs = write_demo(folder, 42)
            for key, file in zip(("places", "reviews"), inputs):
                self.assertEqual(manifest["source_files"][key]["sha256"], hashlib.sha256(file.read_bytes()).hexdigest())
            reviews = records(output, "reviews.csv")
            businesses = records(output, "businesses.csv")
            assigned = records(output, "assignments.csv")
            self.assertEqual(len(businesses), 96)
            self.assertEqual(len(reviews), 3779)
            self.assertEqual(len({b["city"] for b in businesses}), 6)
            self.assertEqual(len({b["category"] for b in businesses}), 4)
            self.assertTrue(all(b["business_id"].startswith("demo-") for b in businesses))
            self.assertFalse({"gPlusUserId", "reviewerName", "reviewText"} & reviews[0].keys())
            self.assertEqual(Counter(r["business_id"] for r in reviews),
                             {b["business_id"]: int(b["review_count"]) for b in businesses})
            self.assertEqual({b["business_id"] for b in assigned}, {b["business_id"] for b in businesses})
            mix = Counter()
            for row in records(output, "category_mix.csv"):
                mix[row["category"]] += int(row["businesses"])
            self.assertEqual(mix, Counter(b["category"] for b in businesses))
            report = (output / "analysis.md").read_text(encoding="utf-8")
            self.assertIn("Synthetic demonstration", report)
            self.assertIn("RQ6", report)
            previous_manifest = (output / "manifest.json").read_bytes()
            again = subprocess.run(command, cwd=ROOT, capture_output=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(previous_manifest, (output / "manifest.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
