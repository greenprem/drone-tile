# farm_tiles v2: tiles anchored on one reference object

```
pip install numpy opencv-python tifffile matplotlib

python farm_tiles.py  AUG_FOLDER  OCT_FOLDER  --ref-image  <aug image>  --other-image  <oct image>
```

`--ref-image` and `--other-image` accept a full path, a file name inside the folder (`DJI_..._0004_D.JPG`), or just the capture number (`0004`). TIFs work too.

## What happens

1. **Picker.** The two images open side by side. Click the **same object** in both.
   - Scroll to zoom, right-drag to pan, click again to move a mark, `r` resets the view.
   - Press **Enter** (or close the window) to accept, or **Esc** to cancel.
   - Pick something **at ground level**: a post where it meets the soil, a stone, a pipe joint, a bed end, a mulch tear.
   - Avoid plant tops and tree crowns. At 15 m altitude, tall things lean in the picture by up to ~0.3 m depending on where they are in the frame.
2. **Tile X.** Tile X = `(0,0)` is centred on that object in both batches. Tile `(+1,0)` is the next tile along axis *i* (roughly "right" in the reference image), `(0,+1)` the next along axis *j* (roughly "down"), and so on. Tiles are measured in metres on the ground.
3. **Offset.** The batch offset is computed from your two clicks and saved in `reference.json` as `other_batch_offset_m` (metres east/north). It is applied to **every frame** of the second batch.
   - It is kept in metres, not pixels: a pixel offset is only true for the two images you clicked, but the ground offset is the same for the whole flight.
   - The next run reuses the saved clicks. Add `--repick` to click again.

## Options

| option | default | meaning |
|---|---|---|
| `--tile W H` | `1.25 2.5` | tile size in metres (W along axis i, H along axis j) |
| `--angle` | `rows` | `rows` = follow the crop rows (auto-detected around your object), `image` = axes of the reference image, or a row azimuth in degrees |
| `--out` | `tiles_out` | output folder |
| `--ref-point U V --other-point U V` | | skip the picker and give the pixels directly |
| `--no-stats` | | only place tiles (skip NDVI) |
| `--no-overlays` | | skip drawing every frame |

## Outputs

| file | what |
|---|---|
| `reference_check.jpg` | **Look at this first.** Tile X and its neighbours in every frame of both batches that sees tile X: the clicked images first, then the others by distance from the frame centre. If the migration works, the red tile sits on the same object in all of them. |
| `overlays/<batch>/*.jpg` | Every frame with all its tiles drawn and labelled. Tile X is red. |
| `tile_placements.csv` | Pixel coordinates (centre + 4 corners) of every tile in every file of every frame: D.JPG and each band TIF, which have their own geometry |
| `tiles.csv`, `tiles.geojson` | Per tile and per batch: NDVI, NDVI of plant pixels only, NDRE, plant cover, and the change between batches. Each tile is measured in the frame where it is closest to the centre. |
| `chips/` | The same tile from both batches side by side |
| `reference.json` | Your clicks, the offset, and the tile grid definition |

## If tiles drift in other frames

Tile X is exact in the two clicked images by construction. In the **other** frames, placement relies on each frame's GPS, height and heading plus your offset. If `reference_check.jpg` shows tile X sliding off the object in some frames, note which frames and by how much. That pattern (always the same frames? one flight line? growing towards the frame edges?) shows what needs correcting next. A second reference object is the natural next step.

---

# tile_stress.py: stressed tiles per date, from your stress profiles

```
python tile_stress.py tiles_out --dap 13aug=10 11oct=70
```

`--dap` gives the plant age (days after transplanting) for each flight. Use any part of the folder name that identifies it uniquely, or two numbers, earlier flight first (`--dap 10 70`). The profiles are loaded automatically from the `profiles/` folder next to the script, and each date gets the one with the nearest `dap`:

| profile | used for | stage |
|---|---|---|
| `profiles/tomato_dap10.json` | 13 Aug (DAP 10) | establishment: seedlings on plastic (NDVI + gaps decide) |
| `profiles/tomato_dap40.json` | flights around DAP 40 | flowering: your original NDVI cuts, NDRE converted (see Calibration) |
| `profiles/tomato_dap70.json` | 11 Oct (DAP 70) | fruiting: closed canopy |

Add a JSON to `profiles/` for any other age. `original/tomato_dap40_original.json` is your profile exactly as you sent it.

Keep it next to `farm_tiles.py`. It reads the tile grid, your reference offset and both batch folders from `tiles_out/reference.json`. Results go to `tiles_out/stress/`.

## What it does

1. **Each date on its own.** Every date is classified separately against the profile for its plant age (DAP). Tiles are **not** compared with each other.
   - **DAP:** from `--dap`, or the flight date (from the photo file names) minus `--planting-date YYYY-MM-DD`.
   - **Which profile:** each date gets the profile whose `"dap"` is nearest (from `profiles/`, or the files given with `--profile`). If the nearest is more than 15 days off (`--max-dap-diff`), that date is **not classified** and the summary says so. To force a profile for one date, use `--profile-for <batch folder name>=<file>`.
2. **Then the two dates are compared.** Tiles classified on both dates are labelled *persistent stress*, *new stress*, *recovered* or *stable*, and every value gets a change column.

## Profile format

This is your old script's format (`original/tomato_dap40_original.json` is your profile exactly as you sent it).

| key | meaning |
|---|---|
| `dap`, `crop`, `stage`, `note` | which plant age the profile is for (`dap` is used to pick the profile) |
| `ndvi` / `ndre` | `stressed_max`, `watch_min`, `watch_max`, `healthy_min`, `bare_max`. A value below `stressed_max` is **stressed**, between `stressed_max` and `healthy_min` is **watch**, at or above `healthy_min` is **healthy**. |
| `stress_rule` | `both`: a tile is stressed only if NDVI **and** NDRE are stressed. `either`: if any one is. `ndvi` / `ndre`: that index only. A tile that isn't stressed but has any index below healthy is **watch**. |
| `prefer_plant_values`, `min_plant_pixels` | use the plant-pixel mean when the tile has at least that many plant pixels, otherwise the whole-tile mean (`values_from` column) |
| `exclude_bare` | a value at or below `bare_max` makes the tile **bare**, not stressed |
| `require_vegetation`, `min_veg_share_pct` | exclude tiles with too little vegetation |
| `require_plastic`, `min_plastic_pixels`, `min_plastic_fraction` | exclude tiles that aren't on plastic-mulch rows |
| `cluster_enabled`, `cluster_distance_m`, `cluster_keep` | stressed tiles closer than this distance form one **hotspot**. Distance is measured edge to edge, so touching tiles are 0 m apart. `cluster_keep` = `most_stressed` / `centroid` / `all` decides where the marker goes. |
| `basemap`, `marker_color`, `map_title` | for `map.html` (`esri_imagery` or `osm`) |

Optional keys added here (defaults shown):

| key | default | meaning |
|---|---|---|
| `plant_ndvi_min` | 0.45 | NDVI above which a pixel counts as a plant pixel |
| `plant_pixels_in` | `"beds"` | plant pixels only on the plastic beds, so furrow weeds don't count; `"tile"` = anywhere in the tile |
| `plastic_from_any_date` | `true` | a closed canopy hides the plastic. With `true`, a tile also passes `require_plastic` if the beds were visible on the other date. On the sample, 20 of 21 October tiles passed only this way. |
| `flag_missing_plants` | `false` (`true` in the shipped profiles) | a tile on a plastic row with fewer than `min_plant_pixels` plant pixels becomes **gap** (missing or dead plants) instead of falling back to whole-tile values and being called *bare*. Gaps are clustered into their own hotspots (`G1`, `G2`…; stressed hotspots are `S1`, `S2`…). |
| `index_calculation` | | text only: records how NDVI/NDRE were calculated when the cuts were set |

**Pixel counts** (`min_plant_pixels`, `min_plastic_pixels`) are in native multispectral pixels: about 7 mm at 15 m, roughly 0.5 cm² each.

## How a tile's values are calculated

1. **Per pixel:** the tile is sampled every 1 cm on the ground, all four bands at the same points, converted to reflectance with DJI's formula (sunlight sensor included). NDVI and NDRE are computed per pixel.
2. **Pixel types:** a plant pixel has NDVI > `plant_ndvi_min` and lies on a bed. A plastic pixel is bright and spectrally flat (NDVI < 0.12).
3. **Tile values:**
   - `plant_ndvi` / `plant_ndre` = mean over the plant pixels.
   - `tile_ndvi` / `tile_ndre` = mean over all pixels: soil, plastic, weeds and plants.
4. **Canopy lean:** on the later date, the bed area is stretched in the direction tall plants lean in the photo (`--canopy-height`, default 0.6 m), so canopy seen from the side still counts.

## Calibration

### What is calibrated, and how

| step | what it does | from |
|---|---|---|
| radiometry | DJI's Mavic 3M formula per band: black level, vignetting, gain × exposure, SensorGainAdjustment, sunlight-sensor Irradiance | photo metadata |
| **photo-position correction** (`--position-correction auto`, on by default) | the same plants read differently at the photo centre and towards the edges (viewing angle, sun hotspot, red-edge vs NIR vignetting). The script compares tiles **inside the same photo** at different distances from the centre, learns the pattern for each date and index, and refers every tile to the photo centre. | your flight (≥ 200 tiles in ≥ 20 photos; otherwise a default for NDRE) |
| NDRE scale | NDRE cuts converted to this calculation (× 0.40) | full-farm `tiles.csv` |
| noise check | tile-to-tile noise per index; warns when a watch band is narrower than 2× the noise, or an index is mostly noise | your flight |

What the full Kiran Mahale `tiles.csv` (1,561 tiles) showed:

- **NDRE rises towards the photo edges, about +0.01 per metre on both dates** (measured between tiles inside the same photo, so it isn't a field pattern). That's 0.04–0.06 at 5 m, more than a whole watch band. NDVI is barely affected. The correction removes it, and the corrected NDRE is flat across the photo.
- **No photo-to-photo errors:** neighbouring tiles in different photos agree as well as neighbours in the same photo, so the sunlight-sensor correction works.
- **Seedling NDRE (DAP 10) is mostly noise:** neighbouring tiles differ by ±0.04, twice the watch band, probably because of the red-edge camera's slight misalignment on 3–8 cm plants. The DAP-10 profile therefore decides on NDVI and gaps only.
- **NDRE scale:** after the correction, plant NDRE = 0.21 × plant NDVI on both dates. Your original DAP-40 profile pairs NDRE = 0.52 × NDVI (0.32 with 0.60), so the old script calculated NDRE differently. The shipped profiles use your NDRE cuts × 0.40 (0.28 → 0.112, 0.32 → 0.128).
- **Effect on your farm** (current profiles → recalibrated):

  | date | before | after |
  |---|---|---|
  | 13 Aug | 612 watch, 507 healthy | 25 watch, 1,095 healthy, 2 gaps |
  | 11 Oct | 359 watch, 821 healthy | 84 watch, 1,096 healthy |

  Most of the earlier "watch" tiles came from the camera, not the plants.

### What is NOT calibrated

The thresholds themselves (what NDVI/NDRE means "stressed" for this tomato at each age) have not been checked against plants. Only field observations can do that. Every run writes **`field_check_<date>.csv`**: about 40–50 tiles to visit (flagged tiles, lowest NDVI/NDRE, tiles near the cuts, healthy references), with GPS and empty columns for what you see (status, symptom, SPAD or other measurement, photo). Visit them within a few days of a flight, fill them in, and the cuts can be fitted to your observations.

## Outputs (`tiles_out/stress/`)

| file | what |
|---|---|
| `summary.txt` | per date: DAP, profile, class counts, why tiles were excluded, NDVI/NDRE against the profile, hotspots; then the change between dates |
| `map.html` | satellite basemap with layers: tile classes, stressed hotspots and gap markers per date, and the change. Needs internet for the map. |
| `tiles.csv` / `tiles.geojson` | every tile, both dates side by side: values (corrected; `*_raw` = before the photo-position correction), plant/plastic pixel counts, values used, per-index class, class and reason, hotspot id, then the change columns |
| `hotspots_<date>.csv` | stressed hotspots (`S…`) and missing-plant gaps (`G…`): marker lat/lon, size, the marker tile's values, all its tiles |
| `transitions.csv` | count of tiles per (earlier class → later class) |
| `field_check_<date>.csv` | tiles to visit in the field, with blank columns for your observations: the way to calibrate the thresholds |
| `maps/classes.png` | static class maps per date and the change map |
| `hotspot_chips/` | per hotspot: the worst tile on each date, plus which pixels were used (green = plant pixels used, orange = other vegetation, white = plastic) |

## Other options

| option | default | use |
|---|---|---|
| `--max-frame-dist` | 6 | tiles only seen further than this (m) from a frame centre are excluded; near frame edges, plants are seen from the side |
| `--canopy-height` | 0.6 | crop height (m) on the later date |
| `--max-dap-diff` | 15 | how far a profile's `dap` may be from the flight's DAP |
| `--chips` | 12 | hotspot pictures per date |
| `--position-correction` | `auto` | `auto` = learn the photo-position pattern per date, `default` = use the built-in NDRE default, `off` = raw values |
