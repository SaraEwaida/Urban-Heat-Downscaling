# urban-heat-downscaling

Estimating 30 m land surface temperature (LST) from roughly 1 km satellite LST, using fine-resolution vegetation, land cover and elevation. First city: Abu Dhabi.

**Status:** milestone 1, data alignment and the input-vs-reference view. Models, multiple dates and the dashboard come next (see [Roadmap](#roadmap)).

## Why

City heat varies block by block, but daily thermal satellites see it at about 1 km. Landsat sees it at about 100 m, but only every 8-16 days. This project learns how to map the daily coarse signal onto the fine urban texture, and measures how well that works on held-out areas and dates.

A note on what is being estimated: this is **land surface temperature** (the skin temperature of roofs, roads, sand and vegetation), not near-surface air temperature at 2 m. The two are related but different. Surface temperature is a key input to street-level air temperature models, which is where this work fits.

## Data

All sources are free and are read from [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) STAC. No account is needed; URLs are signed automatically at read time.

| Layer | Product | Source resolution | Grid in this project | Resampling |
|---|---|---|---|---|
| Coarse input | MODIS Terra MOD11A1 v061 `LST_Day_1km` | 926.6 m | 930 m (31 x 30 m) | nearest |
| Reference | Landsat 8/9 Collection 2 Level-2 `ST_B10` | 30 m grid, **100 m effective thermal** | native 30 m | none |
| Vegetation | NDVI from Landsat SR `red`, `nir08` | 30 m | native 30 m | none |
| Land cover | ESA WorldCover v200 | 10 m | 30 m | nearest (categorical) |
| Elevation | Copernicus DEM GLO-30 | 30 m | 30 m | bilinear |

Units: all temperatures are stored in °C. Scale factors: Landsat ST `DN * 0.00341802 + 149.0` K; MODIS LST `DN * 0.02` K.

Masks:
- Landsat: `QA_PIXEL` fill, cloud, dilated cloud, cirrus, cloud shadow and snow removed; `ST_QA` uncertainty above 3 K removed.
- MODIS: `QC_Day` good quality kept; "other quality" kept only if the LST error class is 2 K or better.

Grids: the fine grid is the Landsat scene's own UTM 40N (EPSG:32640) 30 m grid, cropped to the city box and trimmed so that each 930 m coarse cell holds exactly 31 x 31 fine pixels. Terra (about 10:30 local solar time) is paired with Landsat (about 10:10) on the same day; the MODIS per-pixel view time is stored so the time gap can be reported.

## Setup

```bash
git clone https://github.com/SaraEwaida/urban-heat-downscaling.git
cd urban-heat-downscaling
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Reproduce milestone 1

```bash
uhd fetch configs/abu_dhabi.yaml --list     # list clear same-day Landsat/MODIS pairs
uhd fetch configs/abu_dhabi.yaml            # pick the clearest pair -> data/abu_dhabi/scene.json
uhd align configs/abu_dhabi.yaml            # aligned layers -> data/abu_dhabi/aligned.zarr
uhd preview configs/abu_dhabi.yaml          # figure -> outputs/figures/abu_dhabi_input_vs_reference.png
```

Use `uhd fetch configs/abu_dhabi.yaml --date 2023-07-15` to choose a specific day from the list.

`aligned.zarr` contains:

| Variable | Dims | Meaning |
|---|---|---|
| `lst_ref` | y, x | Landsat surface temperature, °C, NaN where masked |
| `valid_ref` | y, x | Landsat clear and low-uncertainty mask |
| `ndvi` | y, x | NDVI |
| `landcover` | y, x | WorldCover class code (0 = no data) |
| `elevation` | y, x | metres |
| `lst_coarse` | yc, xc | MODIS LST, °C, NaN where masked |
| `valid_coarse` | yc, xc | MODIS QC mask |
| `lst_coarse_on_fine` | y, x | coarse LST repeated onto the 30 m grid (no interpolation) |

Acquisition IDs and times, CRS, transforms, valid fractions and source tiles are kept in the dataset attributes.

## Tests

```bash
pytest
```

The test builds synthetic rasters with the real products' CRSs, grids and encodings (UTM Landsat, sinusoidal MODIS, two geographic WorldCover tiles), runs the full alignment, and checks unit conversion, masks, categorical resampling, coarse/fine nesting, the Zarr round trip and the figure.

## Results

Milestone 1 figure (to be added after running on the real scene):

`outputs/figures/abu_dhabi_input_vs_reference.png`

## Roadmap

1. Aligned rasters and input-vs-reference view for one scene (this milestone).
2. More dates for Abu Dhabi.
3. Splits: a held-out area and held-out dates, assigned before any patching; preprocessing fitted on training data only.
4. Models on the same valid pixels and splits: bilinear interpolation, LightGBM on pixel features, PyTorch U-Net on image patches.
5. Metrics: MAE, RMSE and mean bias in °C, overall and per land-cover class.
6. Streamlit dashboard: linked coarse-input and prediction maps on a shared °C scale, observed temperature and signed error (diverging scale) where reference data exists.

## Layout

```
configs/            city configs (bbox, search window, QA thresholds, paths)
src/uhd/
  products.py       scale factors, fill values, QA bit decoding
  fetch.py          STAC search and Landsat/MODIS pairing -> scene manifest
  align.py          grid definition, warping, masking -> Zarr
  preview.py        input vs reference figure
  cli.py            `uhd` command
tests/              synthetic end-to-end test
```
