# urban-heat-downscaling

Estimating 30 m land surface temperature (LST) from roughly 1 km satellite LST, using fine-resolution vegetation, land cover and elevation. First city: Abu Dhabi.

**Status:** milestone 3, LightGBM on the fixed splits. U-Net and the dashboard come next (see [Roadmap](#roadmap)).

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
uhd fetch configs/abu_dhabi.yaml --date 2023-09-24   # scene used below -> data/abu_dhabi/scene.json
uhd align configs/abu_dhabi.yaml            # aligned layers -> data/abu_dhabi/aligned.zarr
uhd preview configs/abu_dhabi.yaml          # figure -> outputs/figures/abu_dhabi_input_vs_reference.png
```

Without `--date`, `fetch` picks the clearest Landsat scene that has a same-day Terra pass.

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

## Reproduce milestone 2

```bash
uhd align-all configs/abu_dhabi.yaml   # all dates in `dates` on one grid -> data/abu_dhabi/stack.zarr
uhd baseline configs/abu_dhabi.yaml    # bilinear baseline -> outputs/results.csv, outputs/splits.json
```

`align-all` puts every date on the exact pixel grid of the first one and fails if a scene is offset by a fraction of a pixel. Land cover and elevation are stored once; the other layers gain a `time` axis.

### Splits

Defined in the config before any model is fitted, and shared by every method:

| Split | Area | Dates | Use |
|---|---|---|---|
| `train` | training area | training dates | fitting only |
| `test_area` | held-out area | training dates | new place |
| `test_dates` | training area | held-out dates | new day |
| `test_both` | held-out area | held-out dates | new place and new day |

- Held-out area: the eastern 30% of coarse columns. The boundary follows 930 m cell edges, so no input cell is shared between areas. Patch-based models must cut patches inside one split.
- Held-out dates: 2023-08-23 and 2023-09-24, the latest two, so testing is on later conditions than training.
- Scored pixels (the same for every method): clear Landsat, valid MODIS cell, and land (WorldCover water and no-data excluded).
- Errors are prediction minus observation, so a positive bias means too warm.

## Reproduce milestone 3

```bash
uhd lightgbm configs/abu_dhabi.yaml    # train, score every split -> outputs/results.csv
```

Also saves the model (`data/abu_dhabi/models/lightgbm.txt`), predictions for every date (`data/abu_dhabi/predictions/lightgbm.zarr`) and feature importance (`outputs/lightgbm_importance.csv`).

### LightGBM setup

- **Target:** Landsat temperature minus the bilinear surface. Trees cannot extrapolate, so the model learns a correction to the interpolated field rather than absolute temperatures.
- **Features (inputs only):** coarse cell value, bilinear value, the scene's mean coarse temperature, NDVI, elevation, land-cover class, and cell context: mean NDVI in the 930 m cell, the pixel's NDVI and elevation relative to its cell, and the cell's water, built-up, bare and vegetated fractions. Pixel coordinates are not used, so the model cannot memorise places.
- **Data use:** fitted on the `train` split without 2023-06-20; early stopping on 2023-06-20 in the training area. The test splits are only predicted. LightGBM needs no scaling, so there is no other preprocessing to fit.
- NDVI comes from the same Landsat scene as the reference temperature, as is standard for thermal downscaling studies. In operational use it would come from an independent optical image such as Sentinel-2.

## Tests

```bash
pytest
```

The test builds synthetic rasters with the real products' CRSs, grids and encodings (UTM Landsat, sinusoidal MODIS, two geographic WorldCover tiles), runs the full alignment, and checks unit conversion, masks, categorical resampling, coarse/fine nesting, the Zarr round trip and the figure.

## Results

### Milestone 1: aligned scene, Abu Dhabi, 2023-09-24

![Coarse MODIS input vs 30 m Landsat reference](outputs/figures/abu_dhabi_input_vs_reference.png)

- Pair: Landsat 9 `LC09_L2SP_160043_20230924_02_T1` and Terra `MOD11A1.A2023267.h22v06.061`.
- Grid: 713 x 1364 pixels at 30 m (about 21 x 41 km), 23 x 44 coarse cells at 930 m, EPSG:32640.
- Valid pixels: 99.8% of the Landsat reference, 61.9% of the MODIS cells.

Findings that shape the next steps:

- **MODIS gaps are water, not cloud.** Coverage stayed between 59% and 62% across eight clear summer dates. Masked cells are 64% water by WorldCover, against 10% in valid cells; MOD11A1 does not retrieve LST over sea, lagoons and channels. Training and evaluation therefore use land cells valid in both products.
- **Terra passes before Landsat.** On 2023-09-08 the MODIS view time was about 05:58Z against 06:46Z for Landsat, a gap of roughly 50 minutes during morning warming. The view time is stored per scene so the gap can be reported for every date.
- **MODIS reads cooler than Landsat over land**, by roughly 5-8 °C on hot surfaces in the first scenes. Viewing geometry, emissivity assumptions and the time gap all contribute. The models must correct this offset, and mean bias is reported for every method.

### Milestone 2: bilinear baseline, eight dates

All land pixels scored, °C, error = prediction minus Landsat:

| Method | Split | Pixels | MAE | RMSE | Bias |
|---|---|---:|---:|---:|---:|
| Bilinear | train | 1,693,717 | 9.18 | 10.16 | -9.09 |
| Bilinear | test_area | 1,499,384 | 6.12 | 7.09 | -6.01 |
| Bilinear | test_dates | 582,753 | 7.58 | 8.30 | -7.56 |
| Bilinear | test_both | 500,704 | 4.27 | 5.12 | -4.23 |

Per land-cover results are in `outputs/results.csv`.

- **The error is mostly a cold offset.** MAE is close to the absolute bias on every split: interpolated MODIS is 4-9 °C cooler than Landsat. With the mean bias removed, the remaining error is about 3-4.5 °C.
- **The offset depends on place and date.** It is largest in the western training area (island, coast, dense city) and in May-June, and smallest in the eastern desert and suburbs and in August-September. A single constant correction is not enough; the learned models have to explain it from vegetation, land cover, elevation and the coarse temperature.
- **Splits differ in difficulty**, so methods are compared within a split, not splits with each other. The held-out area has nearly as many scored pixels as the training area despite covering 30% of the box, because much of the western part is water and is not scored.

### Milestone 3: LightGBM

To be filled from `outputs/results.csv` after `uhd lightgbm`.

## Roadmap

1. Aligned rasters and input-vs-reference view for one scene.
2. Eight dates for Abu Dhabi, held-out area and dates, bilinear baseline.
3. LightGBM on pixel features, fitted on the training split only (this milestone).
4. PyTorch U-Net on image patches cut inside each split.
5. Comparison of all three methods on every split, overall and per land-cover class.
6. Streamlit dashboard: linked coarse-input and prediction maps on a shared °C scale, observed temperature and signed error (diverging scale) where reference data exists.

## Layout

```
configs/            city configs (bbox, search window, QA thresholds, paths)
src/uhd/
  products.py       scale factors, fill values, QA bit decoding
  fetch.py          STAC search and Landsat/MODIS pairing -> scene manifest
  align.py          grid definition, warping, masking -> Zarr
  preview.py        input vs reference figure
  stack.py          multi-date alignment on one grid
  splits.py         held-out area and dates, shared evaluation mask
  baseline.py       bilinear interpolation baseline
  features.py       per-pixel features with 930 m cell context
  lgbm.py           LightGBM residual model: fit, early stopping, prediction
  metrics.py        MAE, RMSE, bias overall and per land-cover class
  cli.py            `uhd` command
tests/              synthetic end-to-end tests (conftest.py builds the fake products)
```
