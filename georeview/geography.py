"""Conservative country validation against a bundled Natural Earth polygon."""

import json
import re
from pathlib import Path

import numpy as np


class IndiaBoundary:
    def __init__(self):
        self.path = Path(__file__).parent / 'data' / 'india.geojson'
        geometry = json.loads(self.path.read_text(encoding='utf-8'))['geometry']
        polygons = geometry['coordinates'] if geometry['type'] == 'MultiPolygon' else [geometry['coordinates']]
        self.polygons = [[np.asarray(ring, dtype=float) for ring in polygon] for polygon in polygons]

    @staticmethod
    def in_ring(x, y, ring):
        a, b = ring[:-1], ring[1:]
        crosses = (a[:, 1] > y) != (b[:, 1] > y)
        edges = a[crosses]
        ends = b[crosses]
        intersection = edges[:, 0] + (y - edges[:, 1]) * (ends[:, 0] - edges[:, 0]) / (ends[:, 1] - edges[:, 1])
        return bool(np.count_nonzero(x < intersection) % 2)

    def contains(self, lat, lon):
        for polygon in self.polygons:
            shell = polygon[0]
            if not (shell[:, 0].min() <= lon <= shell[:, 0].max() and shell[:, 1].min() <= lat <= shell[:, 1].max()):
                continue
            if self.in_ring(lon, lat, shell) and not any(self.in_ring(lon, lat, hole) for hole in polygon[1:]):
                # Treat coordinates within roughly 1 km of any edge as uncertain.
                for ring in polygon:
                    a, delta = ring[:-1], np.diff(ring, axis=0)
                    length = (delta * delta).sum(axis=1)
                    t = np.clip(((np.array([lon, lat]) - a) * delta).sum(axis=1) / np.maximum(length, 1e-30), 0, 1)
                    if np.min(np.linalg.norm(a + t[:, None] * delta - [lon, lat], axis=1)) <= 0.01:
                        return False
                return True
        return False

    def validate(self, record):
        gps = record.get('gps')
        try:
            lat, lon = map(float, gps)
        except (TypeError, ValueError):
            return None, 'missing_or_invalid_coordinates'
        if not np.isfinite([lat, lon]).all() or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            return None, 'missing_or_invalid_coordinates'
        address = record.get('address') or []
        address = ' '.join(str(x) for x in address) if isinstance(address, list) else str(address)
        if not re.search(r'\bindia\b', address, flags=re.I):
            return None, 'country_address_unconfirmed'
        if re.search(r'\b(nepal|pakistan|bangladesh|bhutan|sri lanka|china|myanmar|united states)\b', address, flags=re.I):
            return None, 'conflicting_country_address'
        if not self.contains(lat, lon):
            return None, 'outside_or_boundary_uncertain'
        return (lat, lon), 'accepted'
