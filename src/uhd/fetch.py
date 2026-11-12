"""Find a same-day Landsat / MODIS daytime pair on Microsoft Planetary Computer and
write a scene manifest (asset hrefs + acquisition metadata).

Nothing large is downloaded here; `uhd align` reads only the needed windows.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

LANDSAT_COLLECTION = "landsat-c2-l2"
LANDSAT_ASSETS = {"st": "lwir11", "qa_pixel": "qa_pixel", "st_qa": "qa", "red": "red", "nir": "nir08"}
MODIS_ASSETS = {"lst": "LST_Day_1km", "qc": "QC_Day", "view_time": "Day_view_time"}
WORLDCOVER_COLLECTION = "esa-worldcover"
DEM_COLLECTION = "cop-dem-glo-30"


def _point_in_ring(x: float, y: float, ring: list) -> bool:
    inside = False
    for (x1, y1), (x2, y2) in zip(ring, ring[1:] + ring[:1]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def bbox_within_geometry(bbox, geometry: dict) -> bool:
    """True if all four bbox corners fall inside the (Multi)Polygon footprint."""
    polys = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
    corners = [(bbox[0], bbox[1]), (bbox[0], bbox[3]), (bbox[2], bbox[1]), (bbox[2], bbox[3])]
    return all(any(_point_in_ring(x, y, [tuple(p[:2]) for p in poly[0]]) for poly in polys) for x, y in corners)


def _client():
    import planetary_computer
    import pystac_client

    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def find_candidates(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Clear Landsat scenes fully covering the bbox, each with a same-day MODIS item."""
    client = _client()
    s = cfg["search"]
    bbox = cfg["bbox"]
    landsat = client.search(
        collections=[LANDSAT_COLLECTION],
        bbox=bbox,
        datetime=f"{s['start']}/{s['end']}",
        query={
            "eo:cloud_cover": {"lt": s["max_landsat_cloud"]},
            "platform": {"in": s["landsat_platforms"]},
        },
    ).item_collection()

    out = []
    for item in sorted(landsat, key=lambda i: i.properties["eo:cloud_cover"]):
        if not bbox_within_geometry(bbox, item.geometry):
            continue
        day = item.datetime.date().isoformat()
        # The collection holds both Terra (MOD11A1, ~10:30) and Aqua (MYD11A1, ~13:30); keep the configured one
        prefix = cfg["coarse"].get("product", "MOD11A1")
        modis = [
            m for m in client.search(collections=[cfg["coarse"]["collection"]], bbox=bbox, datetime=day).items()
            if m.id.startswith(prefix)
        ]
        if not modis:
            continue
        out.append({"landsat": item, "modis": modis[0]})
    return out


def _latest_items(client, collection: str, bbox) -> list:
    items = list(client.search(collections=[collection], bbox=bbox).items())
    if not items:
        raise RuntimeError(f"No {collection} items intersect {bbox}")
    latest = max(i.datetime or datetime.fromisoformat(i.properties["start_datetime"][:19]) for i in items)
    keep = [i for i in items if (i.datetime or datetime.fromisoformat(i.properties["start_datetime"][:19])) == latest]
    return keep


def build_manifest(cfg: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any]:
    client = _client()
    ls, md = pair["landsat"], pair["modis"]
    wc = _latest_items(client, WORLDCOVER_COLLECTION, cfg["bbox"])
    dem = list(client.search(collections=[DEM_COLLECTION], bbox=cfg["bbox"]).items())

    def unsigned(href: str) -> str:
        return href.split("?")[0]

    return {
        "city": cfg["city"],
        "bbox": cfg["bbox"],
        "source": STAC_URL,
        "landsat": {
            "id": ls.id,
            "platform": ls.properties.get("platform"),
            "datetime_utc": ls.datetime.isoformat(),
            "cloud_cover_pct": ls.properties.get("eo:cloud_cover"),
            "epsg": ls.properties.get("proj:epsg") or ls.properties.get("proj:code"),
            "assets": {k: unsigned(ls.assets[v].href) for k, v in LANDSAT_ASSETS.items()},
        },
        "modis": {
            "id": md.id,
            "collection": md.collection_id,
            "date": (md.datetime or datetime.fromisoformat(md.properties["start_datetime"][:19])).date().isoformat(),
            "assets": {k: unsigned(md.assets[v].href) for k, v in MODIS_ASSETS.items()},
        },
        "landcover": {
            "collection": WORLDCOVER_COLLECTION,
            "ids": [i.id for i in wc],
            "hrefs": [unsigned(i.assets["map"].href) for i in wc],
        },
        "dem": {
            "collection": DEM_COLLECTION,
            "ids": [i.id for i in dem],
            "hrefs": [unsigned(i.assets["data"].href) for i in dem],
        },
    }


def fetch(cfg: dict[str, Any], date: str | None = None, list_only: bool = False) -> dict[str, Any] | None:
    cands = find_candidates(cfg)
    if not cands:
        raise RuntimeError("No Landsat/MODIS same-day pairs found; widen the search window or cloud limit.")
    for c in cands:
        ls = c["landsat"]
        print(f"{ls.datetime:%Y-%m-%d %H:%M}Z  {ls.properties['platform']:<10} "
              f"cloud={ls.properties['eo:cloud_cover']:>5.1f}%  {ls.id}  <->  {c['modis'].id}")
    if list_only:
        return None
    if date:
        cands = [c for c in cands if c["landsat"].datetime.date().isoformat() == date]
        if not cands:
            raise RuntimeError(f"No candidate pair on {date}")
    manifest = build_manifest(cfg, cands[0])
    path = Path(cfg["paths"]["scene"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    print(f"Selected {manifest['landsat']['id']} + {manifest['modis']['id']} -> {path}")
    return manifest
