"""Build repeatable fictional inputs in the field layout of the 2018 archive."""

import json
import random
from pathlib import Path


CITIES = (
    ("Bengaluru", "Karnataka", 12.9716, 77.5946),
    ("Delhi", "Delhi", 28.6139, 77.2090),
    ("Hyderabad", "Telangana", 17.3850, 78.4867),
    ("Jaipur", "Rajasthan", 26.9124, 75.7873),
    ("Kolkata", "West Bengal", 22.5726, 88.3639),
    ("Pune", "Maharashtra", 18.5204, 73.8567),
)
SERVICES = (("Café", "Café"), ("Bookshop", "Books"), ("Hotel", "Inn"), ("Restaurant", "Kitchen"))
NAMES = ("Copper Leaf", "Indigo Lane", "Marigold", "Riverbend", "Monsoon", "Banyan Court",
         "Amber Kite", "Rain Tree", "Paper Boat", "Blue Loom", "Mango Grove", "Sunbird")

# Weights for 1–5 stars, review count range, and observed duration in days.
PATTERNS = (
    ((1, 1, 4, 26, 68), (25, 57), (200, 650)),
    ((23, 10, 7, 15, 45), (55, 110), (105, 370)),
    ((35, 36, 15, 10, 4), (17, 42), (250, 700)),
    ((2, 5, 12, 29, 52), (1, 7), (120, 740)),
)


def write_demo(folder: Path, seed: int) -> tuple[Path, Path]:
    rng = random.Random(seed)
    places = folder / "demo-places.jsonl"
    reviews = folder / "demo-reviews.jsonl"

    def line(handle, row):
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    with places.open("w", encoding="utf-8", newline="\n") as p, reviews.open("w", encoding="utf-8", newline="\n") as r:
        for city_no, (city, state, lat, lon) in enumerate(CITIES):
            for service_no, (category, suffix) in enumerate(SERVICES):
                for variant in range(4):
                    index = city_no * 16 + service_no * 4 + variant
                    place_id = f"demo-{index + 1:03d}"
                    name = NAMES[(variant + 4 * service_no + city_no) % len(NAMES)]
                    place = {"gPlusPlaceId": place_id, "name": f"{name} {suffix}",
                             "address": [city, state, "India"], "closed": False,
                             "gps": [round(lat + rng.uniform(-0.012, 0.012), 5),
                                     round(lon + rng.uniform(-0.012, 0.012), 5)]}
                    line(p, place)
                    weights, counts, spans = PATTERNS[(variant + city_no + service_no) % 4]
                    start = 1356998400 + rng.randint(0, 500) * 86400
                    duration = rng.randint(*spans) * 86400
                    for j in range(rng.randint(*counts)):
                        review = {"gPlusPlaceId": place_id, "gPlusUserId": f"fictional-{index}-{j}",
                                  "rating": rng.choices(range(1, 6), weights)[0],
                                  "categories": [category],
                                  "unixReviewTime": start + rng.randint(0, duration)}
                        if index % 13 == 0 and j == 0:
                            review.pop("unixReviewTime")
                        line(r, review)
                        if index == 0 and j == 0:
                            line(r, review)

        # Exercise the place/review join and exclusion audit on every demonstration run.
        for key, address, gps in [
            ("outside", ["Kathmandu", "Nepal"], [27.7172, 85.3240]),
            ("missing-gps", ["Pune", "India"], None),
            ("conflict", ["Kathmandu", "Nepal"], [18.5204, 73.8567]),
        ]:
            line(p, {"gPlusPlaceId": key, "name": "Excluded fixture", "address": address, "gps": gps})
            line(r, {"gPlusPlaceId": key, "rating": 5})
        line(r, {"gPlusPlaceId": "demo-001", "rating": 8})
    return places, reviews
