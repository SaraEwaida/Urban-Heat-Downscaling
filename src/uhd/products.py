"""Product constants: scale factors, fill values, QA bit decoding.

References
- Landsat Collection 2 Level-2 Science Product Guide (LSDS-1619).
- MOD11A1 v061 User Guide (MODIS LST, 1 km daily).
- ESA WorldCover v200 Product User Manual.
"""

from __future__ import annotations

import numpy as np

KELVIN_TO_C = 273.15

# Landsat Collection 2 Level-2
LANDSAT_ST_SCALE = 0.00341802
LANDSAT_ST_OFFSET = 149.0          # K
LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2
LANDSAT_ST_QA_SCALE = 0.01         # K
LANDSAT_FILL = 0
LANDSAT_ST_QA_FILL = -9999
LANDSAT_GRID_M = 30
LANDSAT_TIRS_NATIVE_M = 100        # TIRS / TIRS-2 band 10 native resolution, resampled to 30 m by USGS

# QA_PIXEL bits that make a pixel unusable for surface temperature
QA_FILL, QA_DILATED_CLOUD, QA_CIRRUS, QA_CLOUD, QA_SHADOW, QA_SNOW = 0, 1, 2, 3, 4, 5
QA_WATER = 7

# MOD11A1 v061
MODIS_LST_SCALE = 0.02             # K
MODIS_LST_FILL = 0
MODIS_VIEW_TIME_SCALE = 0.1        # hours, local solar time
MODIS_VIEW_TIME_FILL = 255
MODIS_NATIVE_M = 926.625

# ESA WorldCover classes
WORLDCOVER_NODATA = 0
WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


def _bit(a: np.ndarray, b: int) -> np.ndarray:
    return (a.astype(np.uint32) >> b) & 1 == 1


def landsat_clear(qa_pixel: np.ndarray) -> np.ndarray:
    """True where QA_PIXEL has no fill, cloud, dilated cloud, cirrus, shadow or snow."""
    bad = np.zeros(qa_pixel.shape, dtype=bool)
    for b in (QA_FILL, QA_DILATED_CLOUD, QA_CIRRUS, QA_CLOUD, QA_SHADOW, QA_SNOW):
        bad |= _bit(qa_pixel, b)
    return ~bad


def landsat_st_celsius(dn: np.ndarray) -> np.ndarray:
    out = dn.astype(np.float32) * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET - KELVIN_TO_C
    out[dn == LANDSAT_FILL] = np.nan
    return out


def landsat_sr(dn: np.ndarray) -> np.ndarray:
    out = dn.astype(np.float32) * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET
    out[dn == LANDSAT_FILL] = np.nan
    return out


def landsat_st_uncertainty_k(dn: np.ndarray) -> np.ndarray:
    out = dn.astype(np.float32) * LANDSAT_ST_QA_SCALE
    out[dn == LANDSAT_ST_QA_FILL] = np.nan
    return out


def modis_lst_celsius(dn: np.ndarray) -> np.ndarray:
    out = dn.astype(np.float32) * MODIS_LST_SCALE - KELVIN_TO_C
    out[dn == MODIS_LST_FILL] = np.nan
    return out


def modis_qc_ok(qc: np.ndarray, max_error_k: int = 2) -> np.ndarray:
    """QC_Day: bits 0-1 mandatory QA (0 good, 1 other quality), bits 6-7 LST error class.

    Error class: 0 -> <=1 K, 1 -> <=2 K, 2 -> <=3 K, 3 -> >3 K.
    Keep 'good' pixels, and 'other quality' pixels whose error class is within max_error_k.
    """
    qc = qc.astype(np.uint8)
    mandatory = qc & 0b11
    err_class = (qc >> 6) & 0b11
    err_limit = max(0, min(3, max_error_k - 1))
    return (mandatory == 0) | ((mandatory == 1) & (err_class <= err_limit))
