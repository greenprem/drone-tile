#!/usr/bin/env python3
"""
tile_stress.py - classify every tile of every date against a stress profile for the plants'
age (DAP), then compare the dates.

    python tile_stress.py tiles_out --planting-date 2026-08-06 --profile tomato_dap7.json --profile tomato_dap40.json
    python tile_stress.py tiles_out --dap 13aug=10 11oct=70        (profiles/ next to this script)

Reads the tile grid, the reference offset and both batch folders from tiles_out/reference.json
(the output of farm_tiles.py). Keep this file next to farm_tiles.py.

1. EACH DATE ON ITS OWN, with the profile for its DAP (the nearest 'dap' among the --profile
   files, or the one named with --profile-for):
   - per tile, per pixel (sampled every 1 cm on the ground, DJI-calibrated reflectance):
       plant pixel   = NDVI > plant_ndvi_min (profile, default 0.45) and, with
                       plant_pixels_in = "beds", on the plastic-mulch beds (furrow weeds excluded)
       plastic pixel = bright, spectrally flat (NDVI < 0.12)
   - tile values: plant_ndvi / plant_ndre (mean over plant pixels) and tile_ndvi / tile_ndre
     (mean over all pixels). prefer_plant_values -> plant values when the tile has at least
     min_plant_pixels plant pixels, otherwise the tile means.
   - filters (profile): exclude_bare (value <= bare_max), require_vegetation (plant share >=
     min_veg_share_pct), require_plastic (>= min_plastic_pixels and >= min_plastic_fraction).
   - per index: stressed  value <  stressed_max
                watch     stressed_max <= value < healthy_min
                healthy   value >= healthy_min
   - tile class by stress_rule: "both" = stressed only if NDVI AND NDRE are stressed,
     "either" = if any is, "ndvi" / "ndre" = that index only. Not stressed but with any index
     below healthy -> watch. Otherwise healthy. Or bare / excluded (reason given).
   - hotspots: stressed tiles closer than cluster_distance_m (edge to edge; touching tiles = 0)
     form one hotspot; cluster_keep = most_stressed | centroid | all.
2. THEN COMPARE: for tiles classified on both dates - persistent stress, new stress,
   recovered, stable - plus the change of every value.

Pixel counts (min_plant_pixels, min_plastic_pixels) are in native multispectral pixels
(~7 mm at 15 m), whatever the sampling step.

Outputs (tiles_out/stress/): tiles.csv / tiles.geojson (both dates side by side + change),
hotspots_<date>.csv, transitions.csv, map.html (satellite basemap, layers per date,
hotspot markers, changes), maps/*.png, hotspot_chips/, summary.txt
"""
import argparse, csv, json, math, os, sys, time
import datetime as dt
from collections import defaultdict, Counter
import numpy as np
import cv2
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import farm_tiles as ft  # noqa: E402

BANDS = ft.BANDS
CLASS_COLOURS = {'healthy': '#2e9d4f', 'watch': '#f2c12e', 'stressed': '#d7301f', 'gap': '#7b3294', 'bare': '#8c6d46', 'excluded': '#9e9e9e'}
CLASSIFIED = ('healthy', 'watch', 'stressed', 'gap')
CHANGE_COLOURS = {'persistent stress': '#7f0000', 'new stress': '#ef3b2c', 'recovered': '#2b8cbe',
                  'stable (not stressed)': '#a6d96a'}
PROFILE_DEFAULTS = {
    'stress_rule': 'both', 'prefer_plant_values': True, 'min_plant_pixels': 10, 'exclude_bare': True,
    'require_vegetation': False, 'min_veg_share_pct': 10.0,
    'require_plastic': False, 'min_plastic_pixels': 50, 'min_plastic_fraction': 0.05,
    'plastic_from_any_date': True,      # a closed canopy can hide the plastic: accept beds seen on the other date
    'plant_ndvi_min': 0.45,             # what counts as a plant pixel
    'flag_missing_plants': False,       # plastic row with < min_plant_pixels plant pixels -> class 'gap' (missing/dead plants)
    'plant_pixels_in': 'beds',          # 'beds' (plastic rows only) or 'tile'
    'cluster_enabled': True, 'cluster_distance_m': 1.0, 'cluster_keep': 'most_stressed',
    'basemap': 'esri_imagery', 'marker_color': '#DC2626', 'map_title': '', 'crop': '', 'stage': '', 'note': ''}


def ell(d):
    d = max(1, int(round(d)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


# ------------------------------------------------------------ setup ---------
def load_project(tiles_out):
    ref = json.load(open(os.path.join(tiles_out, 'reference.json')))
    g = ref['grid']
    lat, lon = g['tile_X_center_latlon']
    frame = ft.LocalFrame(lat, lon)
    ai, aj = math.radians(g['axis_i_azimuth_deg']), math.radians(g['axis_j_azimuth_deg'])
    grid = ft.Grid((0.0, 0.0), (math.sin(ai), math.cos(ai)), (math.sin(aj), math.cos(aj)), g['tile_w_m'], g['tile_h_m'])
    folders = {'ref': ref['ref_folder'], 'other': ref['other_folder']}
    names = {k: os.path.basename(os.path.normpath(v)) or k for k, v in folders.items()}
    if names['ref'] == names['other']:
        names = {'ref': 'ref', 'other': 'other'}
    offsets = {'ref': (0.0, 0.0), 'other': (ref['other_batch_offset_m']['dE'], ref['other_batch_offset_m']['dN'])}
    for b, f in folders.items():
        if not os.path.isdir(f):
            sys.exit(f'batch folder not found: {f}  (moved? edit reference.json)')
    caps = {b: ft.discover(f) for b, f in folders.items()}
    return ref, frame, grid, folders, names, offsets, caps


def batch_date(caps_b):
    keys = sorted(caps_b)
    digits = ''.join(ch for ch in keys[0].split('_')[1] if ch.isdigit()) if '_' in keys[0] else ''
    return digits[:8] if len(digits) >= 8 else keys[0]


# ------------------------------------------------------------ profiles ------
def load_profile(path):
    try:
        p = json.load(open(path))
    except Exception as e:
        sys.exit(f'cannot read profile {path}: {e}')
    prof = dict(PROFILE_DEFAULTS); prof.update(p); prof['_file'] = os.path.basename(path)
    for idx in ('ndvi', 'ndre'):
        if idx not in prof:
            if prof['stress_rule'] in (idx, 'both', 'either', 'any'):
                sys.exit(f'{path}: stress_rule "{prof["stress_rule"]}" needs an "{idx}" block')
            continue
        blk = prof[idx]
        for key in ('stressed_max', 'healthy_min'):
            if key not in blk:
                sys.exit(f'{path}: "{idx}" needs "{key}"')
        if blk['stressed_max'] > blk['healthy_min']:
            sys.exit(f'{path}: {idx} stressed_max ({blk["stressed_max"]}) is above healthy_min ({blk["healthy_min"]})')
    if prof['stress_rule'] not in ('both', 'either', 'any', 'ndvi', 'ndre'):
        sys.exit(f'{path}: stress_rule must be both, either, ndvi or ndre')
    if prof['plant_pixels_in'] not in ('beds', 'tile'):
        sys.exit(f'{path}: plant_pixels_in must be "beds" or "tile"')
    if prof['cluster_keep'] not in ('most_stressed', 'centroid', 'all'):
        sys.exit(f'{path}: cluster_keep must be most_stressed, centroid or all')
    return prof


def match_batch(nm, names, batches, opt, notes):
    """Find the batch a user typed: exact, then any unique part of the folder name (ignoring - _ space and
    case), then the closest name (e.g. '110ct' typed for '11oct'), which is reported."""
    norm = lambda x: ''.join(ch for ch in x.lower() if ch.isalnum())
    hit = [b for b in batches if names[b] == nm or b == nm]
    if not hit:
        hit = [b for b in batches if norm(nm) and norm(nm) in norm(names[b])]
    if len(hit) == 1:
        return hit[0]
    import difflib
    score = {b: difflib.SequenceMatcher(None, norm(nm), norm(names[b])).ratio() for b in batches}
    best = sorted(batches, key=lambda b: -score[b])
    if score[best[0]] >= 0.8 and (len(best) == 1 or score[best[0]] - score[best[1]] >= 0.05):
        notes.append(f'{opt}: "{nm}" is not a folder name - using the closest, "{names[best[0]]}"')
        print(f'  {opt}: "{nm}" not found - using the closest folder name "{names[best[0]]}"')
        return best[0]
    sys.exit(f'{opt}: "{nm}" does not match a batch folder. The folders are: {", ".join(names[b] for b in batches)}'
             + (f' - did you mean "{names[best[0]]}"?' if score[best[0]] >= 0.6 else ''))


def assign_profiles(P, names, dates, batches):
    """Return ({batch: profile or None}, {batch: dap or None}, notes)."""
    notes = []
    dap = {b: None for b in batches}
    if P.dap:
        if all('=' in x for x in P.dap):
            for x in P.dap:
                nm, v = x.rsplit('=', 1)
                try:
                    dap[match_batch(nm, names, batches, '--dap', notes)] = int(float(v))
                except ValueError:
                    sys.exit(f'--dap: "{v}" is not a number of days (in {x})')
        elif len(P.dap) == 2 and not any('=' in x for x in P.dap):
            order = sorted(batches, key=lambda b: dates[b])
            dap = dict(zip(order, [int(float(x)) for x in P.dap]))
        else:
            sys.exit('--dap: give NAME=DAP pairs, or two numbers (earlier flight first)')
    elif P.planting_date:
        try:
            p0 = dt.date.fromisoformat(P.planting_date)
        except ValueError:
            sys.exit(f'--planting-date must be YYYY-MM-DD, got {P.planting_date}')
        for b in batches:
            try:
                d = dt.datetime.strptime(dates[b][:8], '%Y%m%d').date()
            except ValueError:
                sys.exit(f'cannot read the flight date of {names[b]} from file names; use --dap')
            dap[b] = (d - p0).days
            if dap[b] < 0:
                sys.exit(f'planting date {p0} is after the {names[b]} flight ({d})')
    prof = {b: None for b in batches}
    explicit = {}
    for item in P.profile_for or []:
        if '=' not in item:
            sys.exit(f'--profile-for needs NAME=FILE, got {item}')
        nm, path = item.split('=', 1)
        explicit[match_batch(nm, names, batches, '--profile-for', notes)] = load_profile(path)
    files = list(P.profile or [])
    if not files and not explicit:
        pdir = P.profiles_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profiles')
        if os.path.isdir(pdir):
            files = sorted(os.path.join(pdir, f) for f in os.listdir(pdir) if f.lower().endswith('.json'))
            if files:
                notes.append(f'profiles from {pdir}: ' + ', '.join(os.path.basename(f) for f in files))
    pool = [load_profile(p) for p in files]
    for b in batches:
        if b in explicit:
            prof[b] = explicit[b]
            pd_ = prof[b].get('dap')
            if dap[b] is not None and pd_ is not None and abs(pd_ - dap[b]) > P.max_dap_diff:
                notes.append(f'{names[b]}: profile {prof[b]["_file"]} is for DAP {pd_}, the flight is at DAP {dap[b]} (used anyway: chosen with --profile-for)')
            continue
        if not pool:
            continue
        if dap[b] is None:
            if len(pool) == 1 and len(batches) == 1:
                prof[b] = pool[0]
            else:
                notes.append(f'{names[b]}: plant age unknown - give --planting-date or --dap so a profile can be picked')
            continue
        cand = [p for p in pool if p.get('dap') is not None]
        if not cand:
            notes.append(f'{names[b]}: none of the profiles has a "dap" field')
            continue
        best = min(cand, key=lambda p: abs(p['dap'] - dap[b]))
        if abs(best['dap'] - dap[b]) > P.max_dap_diff:
            notes.append(f'{names[b]} (DAP {dap[b]}): nearest profile {best["_file"]} is for DAP {best["dap"]} '
                         f'({abs(best["dap"] - dap[b])} days apart) - NOT classified. Add a profile for ~DAP {dap[b]}, '
                         f'or raise --max-dap-diff, or force it with --profile-for {names[b]}={best["_file"]}')
            continue
        prof[b] = best
    return prof, dap, notes


# ------------------------------------------------------------ sampling ------
class Sampler:
    """Samples reflectance on a regular ground raster around tiles (identical points for every date)."""
    def __init__(self, grid, step, margin):
        self.grid, self.st, self.m = grid, step, margin
        self.ns = int(round((grid.W + 2 * margin) / step)); self.nt = int(round((grid.H + 2 * margin) / step))
        s = -margin + (np.arange(self.ns) + 0.5) * step; t = -margin + (np.arange(self.nt) + 0.5) * step
        self.S, self.T = np.meshgrid(s, t)                     # rows = along axis j, cols = along axis i
        self.mp = int(round(margin / step))
        self.inner = (slice(self.mp, self.nt - self.mp), slice(self.mp, self.ns - self.mp))

    def points(self, i, j):
        return self.grid.to_ground(i + self.S / self.grid.W, j + self.T / self.grid.H)

    def sample(self, cam_of, tiles):
        E = np.concatenate([self.points(i, j)[0].ravel() for i, j in tiles])
        N = np.concatenate([self.points(i, j)[1].ravel() for i, j in tiles])
        refl = {}; bad = np.zeros(E.size, bool)
        for band in BANDS:
            c = cam_of(band)
            if c is None:
                continue
            try:
                u, v = c.ground_to_pixel(E, N)
                r, bd = ft.reflectance(ft.remap_points(ft.read_band(c.path), u, v), u, v, c)
            except Exception as e:
                print(f'  skipping {os.path.basename(c.path)}: {e}'); continue
            refl[band] = r; bad |= bd
        if not refl:
            refl = {'G': np.full(E.size, np.nan)}; bad[:] = True
        n = self.nt * self.ns; out = {}
        for k, t in enumerate(tiles):
            sl = slice(k * n, (k + 1) * n)
            out[t] = ({b: r[sl].reshape(self.nt, self.ns) for b, r in refl.items()}, bad[sl].reshape(self.nt, self.ns))
        return out


def indices(R):
    if 'NIR' not in R or 'R' not in R:
        shp = next(iter(R.values())).shape
        nan = np.full(shp, np.nan); return nan, nan, nan
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = (R['NIR'] - R['R']) / (R['NIR'] + R['R'])
        ndre = (R['NIR'] - R['RE']) / (R['NIR'] + R['RE']) if 'RE' in R else np.full_like(ndvi, np.nan)
    bright = sum(R[b] for b in R) / len(R)
    return ndvi, ndre, bright


def bed_map(ndvi, bright, valid, smp, P):
    """Plastic-mulch beds incl. the planting holes (closing + fill), and the visible plastic itself."""
    px = smp.st * 100
    plastic = (ndvi < P.mulch_ndvi) & (bright > P.mulch_bright) & valid
    bed = cv2.morphologyEx(plastic.astype(np.uint8), cv2.MORPH_CLOSE, ell(15 / px))
    bed = ndi.binary_fill_holes(bed) & valid
    bed = cv2.morphologyEx(bed.astype(np.uint8), cv2.MORPH_OPEN, ell(21 / px)).astype(bool)
    return bed, plastic


def count_plants(ndvi, valid, bed, smp, P):
    px = smp.st * 100
    inner_bed = cv2.erode(bed.astype(np.uint8), ell(7 / px)).astype(bool)
    veg = (ndvi > P.seedling_ndvi) & valid & bed
    vegm = cv2.dilate(veg.astype(np.uint8), ell(7 / px)).astype(bool)
    lab, n = ndi.label(vegm); pts = []
    if n:
        ids = np.arange(1, n + 1)
        core = ndi.sum(veg, lab, ids); cy = ndi.center_of_mass(vegm, lab, ids)
        pts = [(y, x) for (y, x), c in zip(cy, core) if c * px * px >= 4 and inner_bed[int(y), int(x)]]
    I = smp.inner
    return sum(1 for y, x in pts if I[0].start <= y < I[0].stop and I[1].start <= x < I[1].stop), pts


def lean_px(cam_, t, grid, smp, h):
    """Canopy tops of height h appear r*h/(H-h) further from the frame centre (raster px: cols, rows)."""
    Ec, Nc = grid.center(*t)
    d = np.array([Ec - cam_.ce, Nc - cam_.cn]) * h / max(cam_.H - h, 1.0)
    return float(d @ grid.ui) / smp.st, float(d @ grid.uj) / smp.st, float(np.hypot(*d))


def plant_zone(bed, smp, P, lean=(0.0, 0.0)):
    """Beds widened by zone_widen and swept along the canopy lean of this frame."""
    z0 = cv2.dilate(bed.astype(np.uint8), ell(2 * P.zone_widen / smp.st))
    acc = z0.copy(); ds, dt_ = lean
    nsteps = int(min(12, max(1, math.ceil(math.hypot(ds, dt_) / 4))))
    for k in range(1, nsteps + 1):
        M = np.float32([[1, 0, ds * k / nsteps], [0, 1, dt_ * k / nsteps]])
        acc |= cv2.warpAffine(z0, M, (z0.shape[1], z0.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    return acc.astype(bool)


def measure(R, bad, smp, plant_ndvi_min, zone, native_px_per_sample):
    """All per-tile values for one date. zone = where plant pixels may be (None = whole tile)."""
    ndvi, ndre, bright = indices(R)
    valid = ~bad & np.isfinite(ndvi)
    I = smp.inner
    v = valid[I]; nd = ndvi[I]; nr = ndre[I]
    veg = (nd > plant_ndvi_min) & v
    plant = veg & (zone[I] if zone is not None else True)
    nv = v.sum()
    m = {'valid_frac': v.mean(),
         'tile_ndvi': float(np.nanmean(nd[v])) if nv else np.nan,
         'tile_ndre': float(np.nanmean(nr[v])) if nv else np.nan,
         'plant_px': int(round(plant.sum() * native_px_per_sample)),
         'plant_ndvi': float(np.nanmean(nd[plant])) if plant.any() else np.nan,
         'plant_ndre': float(np.nanmean(nr[plant])) if plant.any() else np.nan,
         'veg_share_pct': 100.0 * veg.sum() / nv if nv else np.nan}
    return m, ndvi, bright, valid


# ------------------------------------------------------------ classification
def classify(m, prof, bed_frac_other, bed_px_other):
    """Adds value source, per-index classes, tile class and reason to m (in place)."""
    use_plant = prof['prefer_plant_values'] and m['plant_px'] >= prof['min_plant_pixels']
    m['values_from'] = 'plants' if use_plant else 'whole tile'
    val = {'ndvi': m['plant_ndvi'] if use_plant else m['tile_ndvi'],
           'ndre': m['plant_ndre'] if use_plant else m['tile_ndre']}
    m['ndvi_used'], m['ndre_used'] = val['ndvi'], val['ndre']
    m['ndvi_class'] = m['ndre_class'] = ''
    if m['valid_frac'] < 0.5 or not np.isfinite(val['ndvi']):
        m['class'], m['reason'] = 'excluded', 'no usable pixels'; return
    ok_here = m['plastic_px'] >= prof['min_plastic_pixels'] and m['plastic_frac'] >= prof['min_plastic_fraction']
    ok_other = (prof['plastic_from_any_date'] and bed_frac_other is not None and
                bed_px_other >= prof['min_plastic_pixels'] and bed_frac_other >= prof['min_plastic_fraction'])
    m['plastic_ok'] = 'this date' if ok_here else ('beds on other date' if ok_other else 'no')
    on_row = ok_here or ok_other
    if prof['require_plastic'] and not on_row:
        m['class'], m['reason'] = 'excluded', 'no plastic (not a crop row)'; return
    if prof['flag_missing_plants'] and on_row and m['plant_px'] < prof['min_plant_pixels']:
        m['class'] = 'gap'
        m['reason'] = f'plastic row with only {m["plant_px"]} plant pixels (< {prof["min_plant_pixels"]}): missing or dead plants'
        m['severity'] = 1.0 - m['plant_px'] / max(prof['min_plant_pixels'], 1)
        return
    if prof['require_vegetation'] and not (m['veg_share_pct'] >= prof['min_veg_share_pct']):
        m['class'], m['reason'] = 'excluded', f'vegetation < {prof["min_veg_share_pct"]}%'; return
    if prof['exclude_bare']:
        for idx in ('ndvi', 'ndre'):
            if idx in prof and 'bare_max' in prof[idx] and np.isfinite(val[idx]) and val[idx] <= prof[idx]['bare_max']:
                m['class'], m['reason'] = 'bare', f'{idx} {val[idx]:.3f} <= bare_max {prof[idx]["bare_max"]}'; return
    cls = {}
    for idx in ('ndvi', 'ndre'):
        if idx not in prof or not np.isfinite(val[idx]):
            continue
        b = prof[idx]
        cls[idx] = 'stressed' if val[idx] < b['stressed_max'] else ('watch' if val[idx] < b['healthy_min'] else 'healthy')
        m[f'{idx}_class'] = cls[idx]
    rule = prof['stress_rule']
    need = ['ndvi', 'ndre'] if rule in ('both', 'either', 'any') else [rule]
    have = [i for i in need if i in cls]
    if not have:
        m['class'], m['reason'] = 'excluded', f'{"/".join(need)} not measurable'; return
    if rule == 'both':
        stressed = len(have) == 2 and all(cls[i] == 'stressed' for i in have)
    else:
        stressed = any(cls[i] == 'stressed' for i in have)
    if stressed:
        m['class'] = 'stressed'
    elif any(cls[i] != 'healthy' for i in cls):
        m['class'] = 'watch'
    else:
        m['class'] = 'healthy'
    m['reason'] = ', '.join(f'{i} {val[i]:.3f} {cls[i]}' for i in cls) + f' (rule: {rule})'
    # how far below the stressed cut, averaged over the indices the rule uses
    defs = [(prof[i]['stressed_max'] - val[i]) / max(abs(prof[i]['stressed_max']), 1e-6) for i in have]
    m['severity'] = float(np.mean(defs) if rule == 'both' else np.max(defs))


def hotspots(tiles_stressed, m_of, grid, prof, frame):
    """Cluster stressed tiles whose rectangles are closer than cluster_distance_m."""
    ts = sorted(tiles_stressed)
    if not ts:
        return []
    if not prof['cluster_enabled'] or prof['cluster_keep'] == 'all':
        groups = [[t] for t in ts]
    else:
        D = prof['cluster_distance_m']
        gi = max(0, math.ceil(D / grid.W)); gj = max(0, math.ceil(D / grid.H))
        tset = set(ts); seen = set(); groups = []
        def gap(a, b):
            di = max(0, abs(a[0] - b[0]) - 1) * grid.W; dj = max(0, abs(a[1] - b[1]) - 1) * grid.H
            return math.hypot(di, dj)
        for t in ts:
            if t in seen:
                continue
            stack, comp = [t], []; seen.add(t)
            while stack:
                u = stack.pop(); comp.append(u)
                for a in range(-gi - 1, gi + 2):
                    for b in range(-gj - 1, gj + 2):
                        w = (u[0] + a, u[1] + b)
                        if w in tset and w not in seen and gap(u, w) <= D:
                            seen.add(w); stack.append(w)
            groups.append(comp)
    out = []
    for g in groups:
        worst = max(g, key=lambda t: m_of[t].get('severity', 0))
        if prof['cluster_keep'] == 'centroid':
            E = np.mean([grid.center(*t)[0] for t in g]); N = np.mean([grid.center(*t)[1] for t in g])
        else:
            E, N = grid.center(*worst)
        lat, lon = frame.to_ll(E, N)
        out.append({'tiles': g, 'worst': worst, 'lat': lat, 'lon': lon,
                    'severity': float(max(m_of[t].get('severity', 0) for t in g))})
    out.sort(key=lambda h: (-len(h['tiles']), -h['severity']))
    return out


def change_label(a, b):
    ok = ('healthy', 'watch')
    if a == 'stressed' and b == 'stressed':
        return 'persistent stress'
    if a in ok and b == 'stressed':
        return 'new stress'
    if a == 'stressed' and b in ok:
        return 'recovered'
    if a in ok and b in ok:
        return 'stable (not stressed)'
    return ''


# ------------------------------------------------------------ main ----------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('tiles_out', help='output folder of farm_tiles.py (contains reference.json)')
    ap.add_argument('--profile', action='append', help='stress profile JSON with a "dap" field; repeat for several ages')
    ap.add_argument('--profile-for', action='append', metavar='BATCH=FILE', help='use this profile for this batch (folder name)')
    ap.add_argument('--planting-date', help='transplanting date YYYY-MM-DD; DAP = flight date - this')
    ap.add_argument('--dap', nargs='+', metavar='DAP', help='plant age per flight: "13aug=10 11oct=70" '
                    '(folder name, or any unique part of it, = days) or two numbers, earlier flight first')
    ap.add_argument('--profiles-dir', help='folder of profile JSONs used when no --profile is given (default: profiles/ next to this script)')
    ap.add_argument('--max-dap-diff', type=float, default=15, help='use a profile only if its dap is within this many days (default 15)')
    ap.add_argument('--out', help='default: <tiles_out>/stress')
    ap.add_argument('--step-cm', type=float, default=1.0)
    ap.add_argument('--max-frame-dist', type=float, default=6.0, help='exclude tiles only seen further than this (m) from a frame centre')
    ap.add_argument('--canopy-height', type=float, default=0.6, help='crop height (m) on the later date, to follow the canopy lean')
    ap.add_argument('--zone-widen', type=float, default=0.05)
    ap.add_argument('--mulch-ndvi', type=float, default=0.12)
    ap.add_argument('--mulch-bright', type=float, default=0.009)
    ap.add_argument('--seedling-ndvi', type=float, default=0.40, help='for counting seedlings on the bed date')
    ap.add_argument('--chips', type=int, default=12, help='hotspots to illustrate per date')
    ap.add_argument('--position-correction', choices=['auto', 'default', 'off'], default='auto',
                    help='correct values for where the tile sat in its photo: auto = learn per date (fallback: default), off = raw values')
    P = ap.parse_args()
    t0 = time.time()
    ref, frame, grid, folders, names, offsets, caps = load_project(P.tiles_out)
    out = P.out or os.path.join(P.tiles_out, 'stress'); os.makedirs(out, exist_ok=True)
    batches = ['ref', 'other']
    dates = {b: batch_date(caps[b]) for b in batches}
    order = sorted(batches, key=lambda b: dates[b])               # earlier date first
    early, late = order
    prof, dap, notes = assign_profiles(P, names, dates, batches)
    for b in order:
        pr = prof[b]
        print(f'{names[b]}: flight {dates[b]}, DAP {dap[b] if dap[b] is not None else "?"}, '
              + (f'profile {pr["_file"]} (dap {pr.get("dap")}, {pr.get("stage", "")})' if pr else 'NO profile - values only'))
    for n_ in notes:
        print('  ' + n_)

    cams = {}
    def cam(b, key, kind):
        k = (b, key, kind)
        if k not in cams:
            p = caps[b][key].get(kind)
            try:
                cams[k] = ft.Cam(p, frame, offsets[b]) if p else None
            except Exception as e:
                print(f'  skipping {os.path.basename(p)}: {e}'); cams[k] = None
        return cams[k]

    owner = {b: {} for b in batches}
    for b in batches:
        for key in caps[b]:
            c = cam(b, key, 'NIR')
            if c is None:
                continue
            ii, jj, full, _ = grid.tiles_in(c.footprint())
            for i, j in zip(ii[full], jj[full]):
                Ec, Nc = grid.center(i, j); d = math.hypot(Ec - c.ce, Nc - c.cn); t = (int(i), int(j))
                if t not in owner[b] or d < owner[b][t][1]:
                    owner[b][t] = (key, d)
    print('tiles covered: ' + ', '.join(f'{names[b]} {len(owner[b])}' for b in order))

    smp = Sampler(grid, P.step_cm / 100.0, margin=0.40)
    shape = (smp.nt, smp.ns)
    M = {b: {} for b in batches}
    beds, plant_pts = {}, {}

    # ---- earlier date first: it also provides the bed (plastic row) map
    for b in order:
        pr = prof[b] or PROFILE_DEFAULTS
        by = defaultdict(list)
        for t, (key, d) in owner[b].items():
            by[key].append(t)
        for n_, (key, tl) in enumerate(sorted(by.items())):
            cN = cam(b, key, 'NIR')
            gsd = cN.H / float(cN.meta.get('CalibratedFocalLength', cN.K[0, 0]))
            native = (smp.st / gsd) ** 2
            data = smp.sample(lambda band: cam(b, key, band), tl)
            for t, (R, bad) in data.items():
                ndvi, ndre, bright = indices(R)
                valid = ~bad & np.isfinite(ndvi)
                bed_here, plastic = bed_map(ndvi, bright, valid, smp, P)
                if b == early:
                    beds[t] = np.packbits(bed_here)
                if t in beds and pr['plant_pixels_in'] == 'beds':
                    bed = np.unpackbits(beds[t])[:shape[0] * shape[1]].reshape(shape).astype(bool)
                    lean = lean_px(cN, t, grid, smp, P.canopy_height)[:2] if b != early else (0.0, 0.0)
                    zone = plant_zone(bed, smp, P, lean) if bed[smp.inner].mean() > 0.02 else None
                else:
                    zone = None
                m, _, _, _ = measure(R, bad, smp, pr['plant_ndvi_min'], zone, native)
                I = smp.inner
                m['plastic_px'] = int(round(plastic[I].sum() * native))
                m['plastic_frac'] = float(plastic[I].mean())
                m['bed_frac'] = float(bed_here[I].mean())
                m['bed_px'] = int(round(bed_here[I].sum() * native))
                m['plant_pixels_in'] = 'beds' if zone is not None else 'tile'
                m['capture'] = key; m['dist_from_frame_centre_m'] = owner[b][t][1]
                Ec, Nc = grid.center(*t); m['_dE'] = Ec - cN.ce; m['_dN'] = Nc - cN.cn; m['_H'] = cN.H
                if b == early:
                    m['plants_counted'], plant_pts[t] = count_plants(ndvi, valid, bed_here, smp, P)
                M[b][t] = m
            print(f'\r  {names[b]}: {n_ + 1}/{len(by)} frames', end='', flush=True)
        print()

    # ---- frame-position correction (camera/sun geometry), learned per date from tiles in the same photo
    pos_info = {}
    for b in batches:
        other = late if b == early else early
        pos_info[b] = position_correction(M[b], M[other], P)

    # ---- classify each date on its own
    for b in batches:
        other = late if b == early else early
        for t, m in M[b].items():
            if prof[b] is None:
                m['class'], m['reason'] = '', 'no profile for this date'
                continue
            if m['dist_from_frame_centre_m'] > P.max_frame_dist:
                m['class'], m['reason'] = 'excluded', f'only seen > {P.max_frame_dist} m from a frame centre'
                continue
            o = M[other].get(t)
            classify(m, prof[b], o['bed_frac'] if o else None, o['bed_px'] if o else 0)

    # ---- hotspots per date (stressed tiles; gap tiles separately)
    hs, gs = {}, {}
    for b in batches:
        if prof[b] is None:
            continue
        hs[b] = hotspots([t for t, m in M[b].items() if m.get('class') == 'stressed'], M[b], grid, prof[b], frame)
        gs[b] = hotspots([t for t, m in M[b].items() if m.get('class') == 'gap'], M[b], grid, prof[b], frame)
        hid = {t: f'S{k + 1}' for k, h in enumerate(hs[b]) for t in h['tiles']}
        hid.update({t: f'G{k + 1}' for k, h in enumerate(gs[b]) for t in h['tiles']})
        for t, m in M[b].items():
            m['hotspot'] = hid.get(t, '')

    # ---- compare the dates
    common = sorted(set(M[early]) & set(M[late]))
    change = {t: change_label(M[early][t].get('class'), M[late][t].get('class')) for t in common}

    write_outputs(out, grid, frame, names, order, early, late, M, prof, dap, dates, hs, gs, change, P, notes, smp,
                  caps, cam, owner, beds, plant_pts, pos_info)
    print(f'done in {time.time() - t0:.0f} s -> {os.path.abspath(out)}')


# ------------------------------------------------------------ calibration ---
POS_KEYS = ('plant_ndvi', 'plant_ndre', 'tile_ndvi', 'tile_ndre')
POS_DEFAULT = {'plant_ndre': 0.15, 'tile_ndre': 0.15}   # per unit tan(view angle); measured on the Kiran Mahale flights


def position_correction(Mb, Mo, P):
    """Remove the part of each value that depends on where the tile sat in its photo.

    The same plants read differently at the photo centre and towards the edges (viewing angle,
    sun hotspot, band-to-band vignetting differences). Comparing tiles INSIDE the same photo
    (each photo's mean removed) separates this from real field patterns. Model per index:
        value = photo mean + a * r/H + b * east/H + c * north/H
    (r, east, north = tile offset from the camera on the ground, H = flight height). Values are
    referred to the photo centre (r = 0): corrected = value - (a*r + b*east + c*north)/H."""
    info = {}
    on_row = lambda t, m: (m['bed_frac'] >= 0.05 or m['plastic_frac'] >= 0.05 or
                           (t in Mo and Mo[t]['bed_frac'] >= 0.05))
    for k in POS_KEYS:
        rows = [(t, m) for t, m in Mb.items() if np.isfinite(m.get(k, np.nan))]
        for t, m in rows:
            m[k + '_raw'] = m[k]
        if P.position_correction == 'off' or not rows:
            info[k] = None; continue
        fit = [(t, m) for t, m in rows if m['valid_frac'] >= 0.5 and m['dist_from_frame_centre_m'] <= P.max_frame_dist and on_row(t, m)]
        per = Counter(m['capture'] for t, m in fit)
        fit = [(t, m) for t, m in fit if per[m['capture']] >= 3]
        beta, how = None, ''
        if P.position_correction == 'auto' and len(fit) >= 200 and len({m['capture'] for t, m in fit}) >= 20:
            X = np.array([[np.hypot(m['_dE'], m['_dN']) / m['_H'], m['_dE'] / m['_H'], m['_dN'] / m['_H']] for t, m in fit])
            y = np.array([m[k] for t, m in fit]); g = np.array([m['capture'] for t, m in fit])
            keep = np.ones(len(y), bool)
            for _ in range(2):                      # fit, drop outliers (> 3 robust sd), refit
                Xd = X.copy(); yd = y.copy()
                for cap in set(g[keep]):
                    sel = keep & (g == cap)
                    Xd[sel] -= X[sel].mean(0); yd[sel] -= y[sel].mean()
                beta, *_ = np.linalg.lstsq(Xd[keep], yd[keep], rcond=None)
                res = yd - Xd @ beta
                mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
                keep = np.abs(res) <= 3 * max(mad, 1e-6)
            how = f'fitted on {int(keep.sum())} tiles in {len(set(g[keep]))} photos'
        elif P.position_correction != 'off' and k in POS_DEFAULT:
            beta = np.array([POS_DEFAULT[k], 0.0, 0.0])
            how = f'default (only {len(fit)} usable tiles; >= 200 in >= 20 photos needed to fit)'
        if beta is None:
            info[k] = None; continue
        for t, m in rows:
            x = np.array([np.hypot(m['_dE'], m['_dN']), m['_dE'], m['_dN']]) / m['_H']
            m[k] = m[k + '_raw'] - float(x @ beta)
        H = np.median([m['_H'] for t, m in rows])
        info[k] = {'beta': beta, 'how': how, 'at5m': float(beta[0] * 5 / H)}
    return info


def neighbour_scatter(Mb, key):
    """Typical tile-to-tile difference of classified neighbours -> measurement noise (+ fine-scale real variation)."""
    vals = {t: m[key] for t, m in Mb.items() if m.get('class') in CLASSIFIED and np.isfinite(m.get(key, np.nan))}
    same, other = [], []
    for (i, j), v in vals.items():
        for a, b in ((1, 0), (0, 1), (1, 1), (1, -1)):
            w = vals.get((i + a, j + b))
            if w is not None:
                (same if Mb[(i, j)]['capture'] == Mb[(i + a, j + b)]['capture'] else other).append(abs(v - w))
    if len(same) + len(other) < 30:
        return None
    sd = float(np.median(same + other) / (0.6745 * np.sqrt(2)))
    v = np.array(list(vals.values()))
    return {'sd': sd, 'same': float(np.median(same)) if same else np.nan, 'other': float(np.median(other)) if other else np.nan,
            'p5': float(np.percentile(v, 5)), 'p95': float(np.percentile(v, 95))}


def field_check_sheet(out, grid, frame, b, name, Mb, prof, n_each=10):
    """Tiles worth visiting to calibrate the thresholds: the lowest values, tiles near the cuts, and healthy references."""
    cls = [(t, m) for t, m in Mb.items() if m.get('class') in CLASSIFIED and np.isfinite(m.get('ndvi_used', np.nan))]
    if len(cls) < 20:
        return
    pick = {}
    def add(items, why):
        for t, m in items:
            if t not in pick and len([1 for v in pick.values() if v == why]) < n_each:
                pick[t] = why
    add([x for x in cls if x[1]['class'] in ('stressed', 'gap')], 'flagged stressed / gap')
    add(sorted(cls, key=lambda x: x[1]['ndvi_used']), 'lowest NDVI')
    if 'ndre' in prof and any(np.isfinite(m.get('ndre_used', np.nan)) for t, m in cls):
        add(sorted([x for x in cls if np.isfinite(x[1]['ndre_used'])], key=lambda x: x[1]['ndre_used']), 'lowest NDRE')
    cut = prof['ndvi']['stressed_max'] if 'ndvi' in prof else None
    if cut is not None:
        add(sorted(cls, key=lambda x: abs(x[1]['ndvi_used'] - cut)), 'near the NDVI stressed cut')
    rng = np.random.default_rng(7)
    healthy = [x for x in cls if x[1]['class'] == 'healthy']
    add([healthy[k] for k in rng.permutation(len(healthy))], 'healthy reference')
    rows = sorted(pick.items(), key=lambda kv: (kv[0][1], kv[0][0]))
    with open(os.path.join(out, f'field_check_{name}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['visit', 'tile', 'lat', 'lon', 'why_chosen', 'class_by_script', 'ndvi', 'ndre', 'plant_px',
                    'observed_status (healthy/watch/stressed/missing)', 'symptom', 'spad_or_other_measure', 'photo', 'notes'])
        for n_, (t, why) in enumerate(rows):
            m = Mb[t]; lat, lon = frame.to_ll(*grid.center(*t))
            w.writerow([n_ + 1, ft.tid(*t), f'{lat:.8f}', f'{lon:.8f}', why, m['class'], fmt(m['ndvi_used']), fmt(m['ndre_used']),
                        m['plant_px'], '', '', '', '', ''])


# ------------------------------------------------------------ outputs -------
def fmt(v):
    if v is None:
        return ''
    if isinstance(v, (float, np.floating)):
        return f'{v:.4f}' if np.isfinite(v) else ''
    return v


def write_outputs(out, grid, frame, names, order, early, late, M, prof, dap, dates, hs, gs, change, P, notes, smp,
                  caps, cam, owner, beds, plant_pts, pos_info):
    alltiles = sorted(set().union(*[set(M[b]) for b in order]), key=lambda t: (t[1], t[0]))
    keys = ['capture', 'dist_from_frame_centre_m', 'valid_frac', 'plant_pixels_in', 'plant_px', 'plant_ndvi', 'plant_ndre',
            'plant_ndvi_raw', 'plant_ndre_raw',
            'tile_ndvi', 'tile_ndre', 'veg_share_pct', 'plastic_px', 'plastic_frac', 'bed_frac', 'plastic_ok',
            'values_from', 'ndvi_used', 'ndre_used', 'ndvi_class', 'ndre_class', 'class', 'reason', 'hotspot']
    E_, L_ = names[early], names[late]
    rows, feats = [], []
    for t in alltiles:
        Ec, Nc = grid.center(*t); lat, lon = frame.to_ll(Ec, Nc)
        r = {'tile': ft.tid(*t), 'i': t[0], 'j': t[1], 'center_lat': f'{lat:.8f}', 'center_lon': f'{lon:.8f}'}
        for b in order:
            m = M[b].get(t, {})
            r[f'{names[b]}_dap'] = dap[b] if (dap[b] is not None and m) else ''
            for k in keys + (['plants_counted'] if b == early else []):
                r[f'{names[b]}_{k}'] = fmt(m.get(k))
        a, c = M[early].get(t), M[late].get(t)
        for k in ('plant_ndvi', 'plant_ndre', 'tile_ndvi', 'tile_ndre', 'veg_share_pct'):
            r[f'change_{k}'] = fmt(c[k] - a[k]) if (a and c and np.isfinite(a[k]) and np.isfinite(c[k])) else ''
        r['change'] = change.get(t, '')
        r['classes'] = f"{(a or {}).get('class', '-') or '-'} -> {(c or {}).get('class', '-') or '-'}"
        rows.append(r)
        ring = [[round(x, 9) for x in frame.to_ll(e, n)[::-1]] for e, n in grid.corners(*t)]; ring.append(ring[0])
        feats.append({'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [ring]}, 'properties': r})
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(out, 'tiles.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, restval=''); w.writeheader(); w.writerows(rows)
    json.dump({'type': 'FeatureCollection', 'features': feats}, open(os.path.join(out, 'tiles.geojson'), 'w'))

    for b in order:
        if b not in hs:
            continue
        with open(os.path.join(out, f'hotspots_{names[b]}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['hotspot', 'type', 'lat', 'lon', 'tiles', 'area_m2', 'marker_tile', 'ndvi', 'ndre', 'plant_px', 'values_from', 'severity', 'all_tiles'])
            for kind, lst, pre in (('stressed', hs[b], 'S'), ('gap (missing plants)', gs.get(b, []), 'G')):
                for k, h in enumerate(lst):
                    m = M[b][h['worst']]
                    w.writerow([f'{pre}{k + 1}', kind, f"{h['lat']:.8f}", f"{h['lon']:.8f}", len(h['tiles']), round(len(h['tiles']) * grid.W * grid.H, 2),
                                ft.tid(*h['worst']), fmt(m['ndvi_used']), fmt(m['ndre_used']), m['plant_px'], m['values_from'],
                                round(h['severity'], 3), ' '.join(ft.tid(*t) for t in h['tiles'])])

    trans = Counter((M[early][t].get('class') or '-', M[late][t].get('class') or '-') for t in change)
    cl = ['healthy', 'watch', 'stressed', 'gap', 'bare', 'excluded', '-']
    with open(os.path.join(out, 'transitions.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow([f'{E_} \\ {L_}'] + cl)
        for a in cl:
            w.writerow([a] + [trans.get((a, c), 0) for c in cl])

    static_maps(out, grid, order, names, M, hs, change, dap, prof)
    for b in order:
        if prof[b]:
            field_check_sheet(out, grid, frame, b, names[b], M[b], prof[b])
    html_map(out, grid, frame, order, names, M, hs, gs, change, dap, prof, early, late)
    if P.chips:
        hotspot_chips(out, grid, smp, order, names, M, hs, gs, caps, cam, owner, prof, P, early, beds)

    lines = []
    for b in order:
        pr = prof[b]
        lines.append(f'{names[b]}  flight {dates[b]}  DAP {dap[b] if dap[b] is not None else "?"}  '
                     + (f'profile {pr["_file"]} (crop {pr.get("crop", "")}, dap {pr.get("dap")}, stage {pr.get("stage", "")}, rule {pr["stress_rule"]})' if pr else 'no profile'))
        ms = list(M[b].values())
        if not pr:
            onrow = lambda t: any(M[x].get(t, {}).get('bed_frac', 0) >= 0.05 or M[x].get(t, {}).get('plastic_frac', 0) >= 0.05 for x in order)
            crop = [m for t, m in M[b].items() if onrow(t)]
            if crop:
                med = lambda k: np.nanmedian([m[k] for m in crop])
                lines.append(f'   not classified (no profile for {"DAP " + str(dap[b]) if dap[b] is not None else "this date - plant age unknown"}). '
                             f'Measured on {len(crop)} tiles on plastic rows: '
                             f'plant NDVI {med("plant_ndvi"):.3f}, plant NDRE {med("plant_ndre"):.3f}, tile NDVI {med("tile_ndvi"):.3f}, '
                             f'plant share {med("veg_share_pct"):.1f}%')
            continue
        cnt = Counter(m.get('class', '') for m in ms)
        n_cls = sum(cnt[c] for c in CLASSIFIED)
        lines.append('   tiles: ' + ', '.join(f'{c} {cnt[c]}' for c in ('healthy', 'watch', 'stressed', 'gap', 'bare', 'excluded') if cnt[c])
                     + (f'   -> stressed {100 * cnt["stressed"] / n_cls:.1f}% of classified crop tiles' if n_cls else ''))
        reasons = Counter(m['reason'] for m in ms if m.get('class') == 'excluded')
        if reasons:
            lines.append('   excluded because: ' + '; '.join(f'{r} ({n})' for r, n in reasons.most_common()))
        po = Counter(m.get('plastic_ok') for m in ms if m.get('class') in CLASSIFIED and m.get('plastic_ok'))
        if po.get('beds on other date'):
            lines.append(f'   plastic check: {po.get("this date", 0)} tiles show enough plastic on this date, {po["beds on other date"]} passed only '
                         f'because the beds were seen on the other date (canopy hides the plastic; set "plastic_from_any_date": false to disallow)')
        used = [m for m in ms if m.get('class') in ('healthy', 'watch', 'stressed')]
        if used:
            src = Counter(m['values_from'] for m in used)
            med = lambda k: np.nanmedian([m[k] for m in used])
            lines.append(f'   values from: ' + ', '.join(f'{k} {v}' for k, v in src.items())
                         + f'   medians: NDVI {med("ndvi_used"):.3f}, NDRE {med("ndre_used"):.3f} (plant NDVI {med("plant_ndvi"):.3f}, plant NDRE {med("plant_ndre"):.3f}, tile NDVI {med("tile_ndvi"):.3f})')
            if pr:
                for idx in ('ndvi', 'ndre'):
                    if idx in pr:
                        c2 = Counter(m[f'{idx}_class'] for m in used if m.get(f'{idx}_class'))
                        lines.append(f'   {idx.upper()} vs profile (stressed < {pr[idx]["stressed_max"]}, healthy >= {pr[idx]["healthy_min"]}): '
                                     + ', '.join(f'{k} {c2[k]}' for k in ('healthy', 'watch', 'stressed') if c2[k]))
            near = [m['ndre_used'] for m in used if m['dist_from_frame_centre_m'] < 2]
            far = [m['ndre_used'] for m in used if m['dist_from_frame_centre_m'] > 4]
            if len(near) >= 5 and len(far) >= 5 and abs(np.nanmedian(near) - np.nanmedian(far)) > 0.03:
                lines.append(f'   NOTE: NDRE differs with position in the photo (median {np.nanmedian(near):.3f} within 2 m of a frame centre, '
                             f'{np.nanmedian(far):.3f} beyond 4 m) - viewing angle; treat tiles near the thresholds with care')
        pi = pos_info.get(b) or {}
        for k in ('plant_ndvi', 'plant_ndre'):
            if pi.get(k):
                bta = pi[k]['beta']
                lines.append(f'   frame-position correction {k}: {pi[k]["at5m"]:+.3f} at 5 m from the photo centre '
                             f'({pi[k]["how"]}; per unit tan(view angle): radial {bta[0]:+.3f}, east {bta[1]:+.3f}, north {bta[2]:+.3f})')
        for idx in ('ndvi', 'ndre'):
            ns = neighbour_scatter(M[b], f'{idx}_used')
            if not ns:
                continue
            nb = f'neighbours in the same photo differ by {ns["same"]:.3f}' + (
                f', in different photos {ns["other"]:.3f}' if np.isfinite(ns['other']) else '')
            total_sd = max((ns['p95'] - ns['p5']) / 3.29, 1e-9)            # p5-p95 spans 3.29 sd
            share = min(1.0, (ns['sd'] / total_sd) ** 2)
            msg = (f'   {idx.upper()} tile-to-tile noise +-{ns["sd"]:.3f} ({nb}); field spread {ns["p5"]:.3f}-{ns["p95"]:.3f}; '
                   f'~{100 * share:.0f}% of the differences between tiles is noise')
            if idx in pr:
                band = pr[idx]['healthy_min'] - pr[idx]['stressed_max']
                if band < 2 * ns['sd']:
                    msg += f'  - WARNING: the watch band ({band:.3f} wide) is narrower than 2x the noise, tiles near the cuts will flip class'
            if share >= 0.5:
                msg += f'  - WARNING: {idx.upper()} differences between tiles are mostly noise on this date'
            if np.isfinite(ns['other']) and np.isfinite(ns['same']) and ns['other'] > 1.3 * ns['same']:
                msg += '  - WARNING: neighbours in different photos disagree more than in the same photo (photo-to-photo radiometry)'
            lines.append(msg)
        if b in hs:
            lines.append(f'   hotspots: {len(hs[b])} stressed' + (f', {len(gs[b])} missing-plant gaps' if gs.get(b) else '')
                         + (f' (tiles within {pr["cluster_distance_m"]} m merged, marker = {pr["cluster_keep"]})' if pr['cluster_enabled'] else ''))
    lines.append(f'change {E_} -> {L_} (tiles classified on both dates): '
                 + ', '.join(f'{k} {v}' for k, v in Counter(c for c in change.values() if c).most_common()) if any(change.values()) else
                 f'change {E_} -> {L_}: no tile is classified on both dates (both need a profile)')
    lines += [f'note: {n}' for n in notes]
    open(os.path.join(out, 'summary.txt'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def static_maps(out, grid, order, names, M, hs, change, dap, prof):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Patch
    from matplotlib.collections import PatchCollection
    md = os.path.join(out, 'maps'); os.makedirs(md, exist_ok=True)
    allt = sorted(set().union(*[set(M[b]) for b in order]))
    if not allt:
        return
    allc = np.concatenate([grid.corners(*t) for t in allt])
    lim = (allc[:, 0].min() - 2, allc[:, 0].max() + 2, allc[:, 1].min() - 2, allc[:, 1].max() + 2)
    panels = [(names[b], {t: M[b][t].get('class', '') for t in M[b]}, CLASS_COLOURS, b) for b in order if prof[b]]
    if any(change.values()):
        panels.append((f'change {names[order[0]]} -> {names[order[1]]}', change, CHANGE_COLOURS, None))
    if not panels:
        return
    fig, axs = plt.subplots(1, len(panels), figsize=(8 * len(panels), 8), squeeze=False)
    for ax, (title, vals, colours, b) in zip(axs[0], panels):
        ax.set_facecolor('#f4f4f4')
        ax.add_collection(PatchCollection([Polygon(grid.corners(*t)) for t in allt], facecolor='#e2e2e2', edgecolor='none'))
        for k, col in colours.items():
            ts = [t for t, v in vals.items() if v == k]
            if ts:
                ax.add_collection(PatchCollection([Polygon(grid.corners(*t)) for t in ts], facecolor=col, edgecolor='white', lw=0.2))
        if b is not None and b in hs:
            for n_, h in enumerate(hs[b]):
                Ec, Nc = grid.center(*h['worst'])
                ax.plot(Ec, Nc, 'o', mfc='none', mec='black', ms=10, mew=1.5)
                ax.text(Ec, Nc, f' {n_ + 1}', fontsize=8, weight='bold', va='center')
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect('equal')
        sub = f'DAP {dap[b]}, profile {prof[b]["_file"]}' if b is not None else 'tiles classified on both dates'
        ax.set_title(f'{title}\n{sub}', fontsize=10)
        ax.set_xlabel('east (m from tile X)'); ax.set_ylabel('north (m)')
        ax.legend(handles=[Patch(color=c, label=k) for k, c in colours.items()], loc='upper right', fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(md, 'classes.png'), dpi=120); plt.close(fig)
    print('wrote maps/classes.png')


def html_map(out, grid, frame, order, names, M, hs, gs, change, dap, prof, early, late):
    """Self-contained Leaflet page (needs internet for the basemap and the Leaflet library)."""
    layers = []
    for b in order:
        if not prof[b]:
            continue
        feats = []
        for t, m in M[b].items():
            c = m.get('class', '')
            if c in ('', 'excluded'):
                continue
            ring = [[round(x, 9) for x in frame.to_ll(e, n)] for e, n in grid.corners(*t)]
            feats.append({'p': ring, 'c': CLASS_COLOURS.get(c, '#999'),
                          'i': f"tile {ft.tid(*t)} | {names[b]} DAP {dap[b]} | {c} | NDVI {fmt(m['ndvi_used'])} NDRE {fmt(m['ndre_used'])} ({m['values_from']})"})
        marks = []
        for k, h in enumerate(hs.get(b, [])):
            m = M[b][h['worst']]
            marks.append({'ll': [round(h['lat'], 9), round(h['lon'], 9)],
                          'i': f"hotspot S{k + 1} | {names[b]} | {len(h['tiles'])} tile(s) | worst {ft.tid(*h['worst'])}: NDVI {fmt(m['ndvi_used'])} NDRE {fmt(m['ndre_used'])}"})
        gmarks = [{'ll': [round(h['lat'], 9), round(h['lon'], 9)],
                   'i': f"gap G{k + 1} | {names[b]} | {len(h['tiles'])} tile(s) on plastic rows with (almost) no plants"}
                  for k, h in enumerate(gs.get(b, []))]
        layers.append({'name': f'{names[b]} (DAP {dap[b]})', 'tiles': feats, 'marks': marks, 'gmarks': gmarks,
                       'color': prof[b].get('marker_color', '#DC2626')})
    chg = []
    for t, c in change.items():
        if c:
            ring = [[round(x, 9) for x in frame.to_ll(e, n)] for e, n in grid.corners(*t)]
            chg.append({'p': ring, 'c': CHANGE_COLOURS[c], 'i': f'tile {ft.tid(*t)} | {c}'})
    if not layers:
        return
    allll = [frame.to_ll(*grid.center(*t)) for b in order for t in M[b]]
    centre = [float(np.mean([a for a, _ in allll])), float(np.mean([b for _, b in allll]))]
    title = next((prof[b].get('map_title') for b in reversed(order) if prof[b] and prof[b].get('map_title')), 'Stressed tiles')
    basemap = next((prof[b].get('basemap') for b in reversed(order) if prof[b]), 'esri_imagery')
    tiles_url = ('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
                 if basemap == 'esri_imagery' else 'https://tile.openstreetmap.org/{z}/{x}/{y}.png')
    attrib = 'Imagery &copy; Esri' if basemap == 'esri_imagery' else '&copy; OpenStreetMap contributors'
    data = json.dumps({'layers': layers, 'change': chg, 'centre': centre,
                       'change_name': f'change {names[early]} &rarr; {names[late]}', 'legend': CLASS_COLOURS, 'clegend': CHANGE_COLOURS})
    html = f'''<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#map{{height:100%;margin:0;font:13px system-ui,sans-serif}}
.box{{background:#fff;padding:8px 10px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);line-height:1.5}}
.sw{{display:inline-block;width:12px;height:12px;margin-right:6px;vertical-align:-2px;border-radius:2px}}</style></head>
<body><div id="map"></div><script>
const D = {data};
const map = L.map('map', {{maxZoom: 23}}).setView(D.centre, 19);
L.tileLayer('{tiles_url}', {{maxNativeZoom: 19, maxZoom: 23, attribution: '{attrib}'}}).addTo(map);
const over = {{}}; let first = true;
for (const ly of D.layers) {{
  const g = L.layerGroup();
  for (const f of ly.tiles) L.polygon(f.p, {{color: f.c, weight: 0.5, fillColor: f.c, fillOpacity: 0.55}}).bindTooltip(f.i).addTo(g);
  const m = L.layerGroup();
  for (const h of ly.marks) L.circleMarker(h.ll, {{radius: 7, color: '#fff', weight: 2, fillColor: ly.color, fillOpacity: 1}}).bindPopup(h.i).addTo(m);
  over[ly.name + ' - tile classes'] = g; over[ly.name + ' - hotspots (' + ly.marks.length + ')'] = m;
  if (ly.gmarks.length) {{ const gm = L.layerGroup();
    for (const h of ly.gmarks) L.circleMarker(h.ll, {{radius: 7, color: '#fff', weight: 2, fillColor: '{CLASS_COLOURS["gap"]}', fillOpacity: 1}}).bindPopup(h.i).addTo(gm);
    over[ly.name + ' - missing-plant gaps (' + ly.gmarks.length + ')'] = gm; }}
  if (first) {{ g.addTo(map); m.addTo(map); first = false; }}
}}
if (D.change.length) {{
  const g = L.layerGroup();
  for (const f of D.change) L.polygon(f.p, {{color: f.c, weight: 0.5, fillColor: f.c, fillOpacity: 0.6}}).bindTooltip(f.i).addTo(g);
  over[D.change_name] = g;
}}
L.control.layers(null, over, {{collapsed: false}}).addTo(map);
const all = []; for (const ly of D.layers) for (const f of ly.tiles) all.push(...f.p);
if (all.length) map.fitBounds(L.latLngBounds(all), {{padding: [30, 30], maxZoom: 22}});
const lg = L.control({{position: 'bottomleft'}});
lg.onAdd = () => {{ const d = L.DomUtil.create('div', 'box');
  d.innerHTML = '<b>{title}</b><br>' + Object.entries(D.legend).filter(([k]) => k != 'excluded').map(([k, c]) => `<span class="sw" style="background:${{c}}"></span>${{k}}`).join('<br>')
    + '<br><i>change</i><br>' + Object.entries(D.clegend).map(([k, c]) => `<span class="sw" style="background:${{c}}"></span>${{k}}`).join('<br>');
  return d; }};
lg.addTo(map);
</script></body></html>'''
    open(os.path.join(out, 'map.html'), 'w').write(html)
    print('wrote map.html')


def hotspot_chips(out, grid, smp, order, names, M, hs, gs, caps, cam, owner, prof, P, early, beds):
    od = os.path.join(out, 'hotspot_chips'); os.makedirs(od, exist_ok=True)
    shape = (smp.nt, smp.ns); mp = smp.mp; n = 0
    for b, pre, k, h in [(b, pre, k, h) for b in order for pre, src in (('S', hs), ('G', gs))
                         for k, h in enumerate(src.get(b, [])[:P.chips])]:
        if True:
            t = h['worst']; pans = []
            for bb in order:
                if t not in owner[bb]:
                    continue
                key = owner[bb][t][0]; c = cam(bb, key, 'RGB') or cam(bb, key, 'NIR')
                E, N = smp.points(*t); u, v = c.ground_to_pixel(E, N)
                chip = ft.auto_level(cv2.remap(ft.read_display(c.path), u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR))
                cv2.rectangle(chip, (mp, mp), (chip.shape[1] - mp, chip.shape[0] - mp), (0, 255, 255), 1)
                m = M[bb][t]
                lab = f"{names[bb]}: {m.get('class') or '-'}  NDVI {fmt(m['ndvi_used'])} NDRE {fmt(m['ndre_used'])}"
                pans.append(ft.label_bar(chip, lab, 24))
            key = owner[b][t][0]
            R, bad = smp.sample(lambda band: cam(b, key, band), [t])[t]
            ndvi, ndre, bright = indices(R); valid = ~bad & np.isfinite(ndvi)
            pr = prof[b]
            veg = (ndvi > pr['plant_ndvi_min']) & valid
            _, plastic = bed_map(ndvi, bright, valid, smp, P)
            zone = None
            if t in beds and pr['plant_pixels_in'] == 'beds':
                bed = np.unpackbits(beds[t])[:shape[0] * shape[1]].reshape(shape).astype(bool)
                lean = lean_px(cam(b, key, 'NIR'), t, grid, smp, P.canopy_height)[:2] if b != early else (0.0, 0.0)
                zone = plant_zone(bed, smp, P, lean)
            viz = np.full(shape + (3,), 40, np.uint8); viz[plastic] = (235, 235, 235)
            viz[veg] = (0, 140, 255)
            viz[veg & (zone if zone is not None else True)] = (40, 190, 40)
            if zone is not None:
                e = cv2.morphologyEx(zone.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool); viz[e] = (255, 0, 255)
            cv2.rectangle(viz, (mp, mp), (viz.shape[1] - mp, viz.shape[0] - mp), (0, 255, 255), 1)
            pans.append(ft.label_bar(viz, 'green=plant px used, orange=other veg, white=plastic', 24))
            row = np.hstack(sum([[p, np.full((p.shape[0], 6, 3), 255, np.uint8)] for p in pans], [])[:-1])
            kind = 'hotspot' if pre == 'S' else 'gap'
            row = ft.label_bar(row, f'{names[b]} {kind} {pre}{k + 1}: {len(h["tiles"])} tile(s), worst {ft.tid(*t)} - {M[b][t]["reason"]}', 28)
            cv2.imwrite(os.path.join(od, f'{names[b]}_{kind}_{pre}{k + 1:02d}.jpg'), row, [cv2.IMWRITE_JPEG_QUALITY, 90]); n += 1
    if n:
        print(f'wrote hotspot_chips/ ({n})')


if __name__ == '__main__':
    main()
