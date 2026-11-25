"""Stream legacy Google Local records into privacy-minimized analytical tables."""

import ast
import csv
import gzip
import hashlib
import json
import math
import re
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .geography import IndiaBoundary


CITIES = {
    'Bengaluru': (12.9716, 77.5946, ('bengaluru', 'bangalore')),
    'Delhi': (28.6139, 77.2090, ('delhi', 'new delhi')),
    'Hyderabad': (17.385, 78.4867, ('hyderabad',)),
    'Jaipur': (26.9124, 75.7873, ('jaipur',)),
    'Kolkata': (22.5726, 88.3639, ('kolkata', 'calcutta')),
    'Pune': (18.5204, 73.8567, ('pune',)),
    'Mumbai': (19.076, 72.8777, ('mumbai', 'bombay')),
    'Chennai': (13.0827, 80.2707, ('chennai', 'madras')),
    'Ahmedabad': (23.0225, 72.5714, ('ahmedabad',)),
    'Lucknow': (26.8467, 80.9462, ('lucknow',)),
}


def city_name(record, lat, lon):
    address = str(record.get('address', '')).lower()
    for name, (y, x, aliases) in CITIES.items():
        if abs(y - lat) <= 0.5 and abs(x - lon) <= 0.5 and any(re.search(r'\b' + alias + r'\b', address) for alias in aliases):
            return name
    return 'Unknown city'


def read_records(path, counts, prefix):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8-sig') as stream:
        for line in stream:
            counts[prefix + '_lines'] += 1
            try:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = ast.literal_eval(line)
                if not isinstance(record, dict):
                    raise ValueError('Expected dictionary')
            except (ValueError, SyntaxError, TypeError):
                counts[prefix + '_malformed'] += 1
                continue
            if 'gmap_id' in record:
                raise ValueError('The gmap_id schema belongs to the excluded 2021 release. Supply the 2018 archives.')
            yield record


def review_date(value):
    try:
        stamp = float(value)
        date = datetime.fromtimestamp(stamp, timezone.utc)
        if 1990 <= date.year <= 2018:
            return date.date().isoformat()
    except (ValueError, TypeError, OSError, OverflowError):
        pass
    return None


def prepare(places_path, reviews_path, output_dir, *, sentiment=False):
    if sentiment:
        raise ValueError('Text sentiment is unavailable until a language-validated processor is configured. Omit --sentiment to use rating evidence.')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    boundary = IndiaBoundary()
    counts = Counter()
    places, fingerprints, conflicts = {}, {}, set()
    for record in read_records(places_path, counts, 'places'):
        identity = str(record.get('gPlusPlaceId') or '')
        if not identity:
            counts['missing_business_id'] += 1
            continue
        signature = json.dumps({k: record.get(k) for k in ('name', 'address', 'gps', 'closed')}, sort_keys=True)
        fingerprint = hashlib.sha256(signature.encode()).digest()
        if identity in fingerprints:
            counts['duplicate_place_rows'] += 1
            if fingerprints[identity] != fingerprint:
                conflicts.add(identity)
                places.pop(identity, None)
            continue
        fingerprints[identity] = fingerprint
        coordinates, reason = boundary.validate(record)
        counts['places_' + reason] += 1
        if coordinates is None:
            continue
        if record.get('closed') is True:
            counts['closed_businesses'] += 1
            continue
        lat, lon = coordinates
        places[identity] = {'business_id': identity, 'name': str(record.get('name') or 'Unnamed business'),
                            'city': city_name(record, lat, lon), 'latitude': lat, 'longitude': lon,
                            'counts': np.zeros(5, dtype=np.int64), 'dates': [], 'categories': Counter()}
    counts['conflicting_place_ids'] = len(conflicts)
    counts['eligible_places'] = len(places)
    del fingerprints
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        database = sqlite3.connect(str(Path(temporary) / 'duplicates.sqlite'))
        try:
            database.execute('CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID')
            with (out / 'reviews.csv').open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['business_id', 'rating', 'review_date'])
                writer.writeheader()
                for record in read_records(reviews_path, counts, 'reviews'):
                    identity = str(record.get('gPlusPlaceId') or '')
                    place = places.get(identity)
                    if place is None:
                        counts['reviews_ineligible_place'] += 1
                        continue
                    try:
                        rating = float(record.get('rating'))
                        if not math.isfinite(rating) or rating != int(rating) or not 1 <= rating <= 5:
                            raise ValueError('Invalid stars')
                    except (TypeError, ValueError):
                        counts['reviews_invalid_rating'] += 1
                        continue
                    date = review_date(record.get('unixReviewTime'))
                    # Exact event deduplication includes identity and raw text only in a transient digest.
                    key = [identity, record.get('gPlusUserId'), rating, record.get('unixReviewTime'), record.get('reviewText')]
                    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).digest()
                    inserted = database.execute('INSERT OR IGNORE INTO seen VALUES (?)', (digest,)).rowcount
                    if not inserted:
                        counts['reviews_duplicate'] += 1
                        continue
                    counts['reviews_retained'] += 1
                    if date is None:
                        counts['reviews_without_valid_date'] += 1
                    else:
                        place['dates'].append(date)
                    place['counts'][int(rating) - 1] += 1
                    categories = record.get('categories') or []
                    if isinstance(categories, str):
                        categories = [categories]
                    place['categories'].update(str(c).strip() for c in categories if str(c).strip())
                    writer.writerow({'business_id': identity, 'rating': int(rating), 'review_date': date or ''})
        finally:
            database.close()
    businesses = []
    for identity in sorted(places):
        place = places[identity]
        stars = place.pop('counts')
        n = int(stars.sum())
        if not n:
            counts['places_without_eligible_reviews'] += 1
            continue
        dates = place.pop('dates')
        categories = place.pop('categories')
        mean = float(stars @ np.arange(1, 6) / n)
        first, last = (min(dates), max(dates)) if dates else (None, None)
        span = (datetime.fromisoformat(last) - datetime.fromisoformat(first)).days if dates else 0
        place.update(category=sorted(categories, key=lambda c: (-categories[c], c))[0] if categories else 'Unknown category',
                     review_count=n, mean_rating=mean,
                     rating_variance=float(stars @ (np.arange(1, 6) - mean)**2 / n) if n > 1 else None,
                     positive_ratio=float(stars[3:].sum() / n), negative_ratio=float(stars[:2].sum() / n),
                     dated_review_count=len(dates), reviews_per_month=len(dates) / (span / 30.4375) if span >= 30 else None,
                     first_review_date=first, last_review_date=last, sentiment_mean=None, sentiment_coverage=0.0)
        place.update({f'star_{i+1}_ratio': float(value / n) for i, value in enumerate(stars)})
        businesses.append(place)
    columns = ['business_id', 'name', 'city', 'latitude', 'longitude', 'category', 'review_count', 'mean_rating',
               'rating_variance', 'positive_ratio', 'negative_ratio', 'dated_review_count', 'reviews_per_month',
               'first_review_date', 'last_review_date', 'sentiment_mean', 'sentiment_coverage',
               *(f'star_{i}_ratio' for i in range(1, 6))]
    pd.DataFrame(businesses, columns=columns).to_csv(out / 'businesses.csv', index=False)
    counts['businesses_retained'] = len(businesses)
    audit = {'counts': dict(counts), 'boundary_sha256': hashlib.sha256(boundary.path.read_bytes()).hexdigest(),
             'geography_policy': 'Natural Earth 1:10m India polygon, 0.01 degree edge exclusion and explicit India address required.',
             'city_policy': 'Address alias corroborated within 0.5 degrees of one of ten city centers; other eligible places use Unknown city.',
             'frequency_policy': 'Dated review count divided by first-to-last span in months; unavailable for spans under 30 days.',
             'sentiment_policy': 'Unavailable pending a validated language-specific processor.',
             'duplicate_policy': 'Identical business, reviewer, rating, timestamp and text tuple. Conflicting place metadata excludes the place ID.'}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return audit
