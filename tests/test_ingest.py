import json
import tempfile
import unittest
from pathlib import Path

from georeview.geography import IndiaBoundary
from georeview.ingest import prepare


class IngestionTests(unittest.TestCase):
    def test_country_and_boundary_checks(self):
        boundary = IndiaBoundary()
        self.assertTrue(boundary.contains(18.5204, 73.8567))
        self.assertFalse(boundary.contains(27.7172, 85.3240))
        self.assertIsNone(boundary.validate({'gps': [18.5204, 73.8567], 'address': ['Nepal']})[0])
        self.assertIsNone(boundary.validate({'address': ['India']})[0])

    def test_conflicting_place_ids_and_2021_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            places, reviews = root / 'places.jsonl', root / 'reviews.jsonl'
            good = {'gPlusPlaceId': 'x', 'name': 'Fixture', 'gps': [18.5204, 73.8567], 'address': ['Pune', 'India']}
            bad = {**good, 'gps': [27.7172, 85.3240], 'address': ['Nepal']}
            places.write_text(json.dumps(good) + '\n' + json.dumps(bad), encoding='utf-8')
            reviews.write_text(json.dumps({'gPlusPlaceId': 'x', 'rating': 5}), encoding='utf-8')
            audit = prepare(places, reviews, root / 'out')
            self.assertEqual(audit['counts']['businesses_retained'], 0)
            self.assertEqual(audit['counts']['conflicting_place_ids'], 1)
            places.write_text(json.dumps({'gmap_id': '2021-record'}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, '2021'):
                prepare(places, reviews, root / 'invalid')
