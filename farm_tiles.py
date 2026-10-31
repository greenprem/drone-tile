#!/usr/bin/env python3
"""
farm_tiles.py v2 - tiles anchored on ONE manually picked reference object (DJI Mavic 3M).

    python farm_tiles.py REF_FOLDER OTHER_FOLDER --ref-image IMG --other-image IMG [options]

1. A picker opens with the two images side by side. Click the SAME physical object in
   both (scroll = zoom, right-drag = pan, click again to move the mark, Enter = done).
2. Tile X = (0,0) is centred on that object in both batches. X+1, X+2, ... continue from
   it in metres on the ground, so the same tile id is the same patch of field everywhere.
3. The offset between the batches is computed from the two clicks, saved to
   reference.json, and applied to every frame of the other batch. It is stored in metres
   on the ground, not in pixels: a pixel offset is only valid for the two clicked images,
   whereas a ground offset holds for every frame of the flight.

Outputs (in --out):
  reference.json         clicks, offset, grid definition (re-used on the next run)
  reference_check.jpg    tile X and its neighbours in EVERY frame that sees tile X
  overlays/<batch>/      every frame with its tiles drawn (tile X in red)
  tile_placements.csv    pixel corners of every tile in every file (JPG + 4 TIFs)
  tiles.csv / .geojson   per-tile NDVI/NDRE/cover for both batches and the change
  chips/                 the same tile from both batches side by side

Requires: numpy, opencv-python, tifffile, matplotlib
"""
import argparse, csv, json, math, os, re, sys, random, time
from collections import defaultdict
import numpy as np
import cv2

R_EARTH = 6378137.0
OUTSIDE = -1e5          # pixel value used for ground points that are not in a frame
BANDS = ('G', 'R', 'RE', 'NIR')
FILE_KINDS = ('RGB',) + BANDS


# ----------------------------------------------------------------- metadata --
def read_xmp(path):
    with open(path, 'rb') as f:
        b = f.read(300000)
    i = b.find(b'<x:xmpmeta'); j = b.find(b'</x:xmpmeta>')
    if i < 0:
        raise ValueError(f'no XMP metadata in {path}')
    return dict(re.findall(r'(?:drone-dji|tiff|exif|Camera):(\w+)="([^"]*)"', b[i:j].decode('utf8', 'ignore')))


class LocalFrame:
    """Local east/north metres around an origin (equirectangular; mm-accurate over a farm)."""
    def __init__(self, lat0, lon0):
        self.lat0, self.lon0 = lat0, lon0
        self.kx = math.radians(1) * R_EARTH * math.cos(math.radians(lat0))
        self.ky = math.radians(1) * R_EARTH

    def to_en(self, lat, lon):
        return (lon - self.lon0) * self.kx, (lat - self.lat0) * self.ky

    def to_ll(self, e, n):
        return self.lat0 + n / self.ky, self.lon0 + e / self.kx


class Cam:
    """Ground <-> pixel model of one file (D.JPG or one MS band TIF) of one capture.

    Flat ground, nadir camera, RTK position, height above take-off, gimbal yaw, DJI lens
    calibration (DewarpData) and, for MS bands, DJI's CalibratedHMatrix band alignment.
    `offset` (dE, dN) is the batch offset: true ground = metadata ground + offset."""
    def __init__(self, path, frame, offset=(0.0, 0.0), rgb_pp_shift=(-50, -20)):
        m = read_xmp(path)
        self.path, self.meta = path, m
        self.is_rgb = path.upper().endswith(('.JPG', '.JPEG'))
        self.H = float(m['RelativeAltitude'])
        yaw = float(m['GimbalYawDegree']); roll = float(m['GimbalRollDegree'])
        if abs(abs(roll) - 180) < 1:          # pitch -90 gimbal lock: (yaw, 180) == (yaw+180, 0)
            yaw += 180
        self.yaw = yaw % 360
        pitch = float(m.get('GimbalPitchDegree', -90))
        self.nadir_ok = abs(pitch + 90) <= 3
        fx, fy, cxo, cyo, k1, k2, p1, p2, k3 = map(float, m['DewarpData'].split(';')[1].split(','))
        W, Hh = (5280, 3956) if self.is_rgb else (2592, 1944)
        self.size = (W, Hh)
        self.ocx = float(m.get('CalibratedOpticalCenterX', W / 2))
        self.ocy = float(m.get('CalibratedOpticalCenterY', Hh / 2))
        cx = self.ocx + cxo + (rgb_pp_shift[0] if self.is_rgb else 0)
        cy = self.ocy + cyo + (rgb_pp_shift[1] if self.is_rgb else 0)
        self.K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], float)
        self.Kinv = np.linalg.inv(self.K)
        self.dist = np.array([k1, k2, p1, p2, k3], float)
        # The radial distortion polynomial folds back beyond some radius: points far outside the
        # field of view would wrap around INTO the image. Find where it stops being monotonic.
        rr = np.linspace(0, 3, 3001); f = rr * (1 + k1 * rr ** 2 + k2 * rr ** 4 + k3 * rr ** 6)
        bad = np.where(np.diff(f) <= 0)[0]
        self.r_valid = 0.97 * rr[bad[0]] if len(bad) else 3.0
        self.Hm = None
        if 'CalibratedHMatrix' in m:
            self.Hm = np.array(list(map(float, m['CalibratedHMatrix'].split(','))), float).reshape(3, 3)
            self.Hmi = np.linalg.inv(self.Hm)
            f = float(m['CalibratedFocalLength'])
            self.Kid = np.array([[f, 0, self.ocx], [0, f, self.ocy], [0, 0, 1]], float)
            self.Kidi = np.linalg.inv(self.Kid)
        ce, cn = frame.to_en(float(m['GpsLatitude']), float(m['GpsLongitude']))
        self.ce, self.cn = ce + offset[0], cn + offset[1]
        t = math.radians(self.yaw)
        self.fwd = np.array([math.sin(t), math.cos(t)])        # image "up" on the ground
        self.right = np.array([math.cos(t), -math.sin(t)])     # image "right" on the ground

    def ground_to_pixel(self, E, N):
        E = np.asarray(E, float); N = np.asarray(N, float); shp = E.shape
        de, dn = E.ravel() - self.ce, N.ravel() - self.cn
        xr = (de * self.right[0] + dn * self.right[1]) / self.H
        yf = (de * self.fwd[0] + dn * self.fwd[1]) / self.H
        pts = np.stack([xr, -yf, np.ones_like(xr)], 1)
        if self.Hm is not None:
            p = (self.Hmi @ (self.Kid @ pts.T)).T
            pts = (self.Kinv @ (p / p[:, 2:3]).T).T
        # OpenCV's lens model (k1,k2,p1,p2,k3), written out in numpy: same result as
        # cv2.projectPoints but ~20x faster on millions of points
        x = pts[:, 0] / pts[:, 2]; y = pts[:, 1] / pts[:, 2]
        k1, k2, p1, p2, k3 = self.dist
        r2 = x * x + y * y; rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        uv = np.stack([self.K[0, 0] * xd + self.K[0, 2], self.K[1, 1] * yd + self.K[1, 2]], 1)
        far = np.sqrt(r2) > self.r_valid
        uv[far] = OUTSIDE                      # beyond the lens model's valid range -> not in this image
        return uv[:, 0].reshape(shp), uv[:, 1].reshape(shp)

    def pixel_to_ground(self, u, v):
        u = np.asarray(u, float); v = np.asarray(v, float); shp = u.shape
        nrm = cv2.undistortPoints(np.stack([u.ravel(), v.ravel()], 1).reshape(-1, 1, 2), self.K, self.dist).reshape(-1, 2)
        if self.Hm is not None:
            q = self.Hm @ (self.K @ np.c_[nrm, np.ones(len(nrm))].T)
            nrm = (self.Kidi @ (q / q[2]))[:2].T
        right = nrm[:, 0] * self.H; fwd = -nrm[:, 1] * self.H
        E = self.ce + right * self.right[0] + fwd * self.fwd[0]
        N = self.cn + right * self.right[1] + fwd * self.fwd[1]
        return E.reshape(shp), N.reshape(shp)

    def footprint(self, inset_px=None, n=24):
        W, Hh = self.size
        a = (40 if not self.is_rgb else 20) if inset_px is None else inset_px
        us = np.r_[np.linspace(a, W - a, n), np.full(n, W - a), np.linspace(W - a, a, n), np.full(n, a)]
        vs = np.r_[np.full(n, a), np.linspace(a, Hh - a, n), np.full(n, Hh - a), np.linspace(Hh - a, a, n)]
        return np.stack(self.pixel_to_ground(us, vs), 1)


def points_in_poly(px, py, poly):
    x = np.asarray(px, float); y = np.asarray(py, float)
    inside = np.zeros(x.shape, bool)
    xs, ys = poly[:, 0], poly[:, 1]; j = len(poly) - 1
    for i in range(len(poly)):
        inside ^= ((ys[i] > y) != (ys[j] > y)) & (x < (xs[j] - xs[i]) * (y - ys[i]) / ((ys[j] - ys[i]) + 1e-12) + xs[i])
        j = i
    return inside


def remap_points(img, u, v):
    n = u.size; w = 1024; h = max(1, -(-n // w)); pad = h * w - n
    mu = np.r_[u.ravel(), np.full(pad, -1e4)].astype(np.float32).reshape(h, w)
    mv = np.r_[v.ravel(), np.full(pad, -1e4)].astype(np.float32).reshape(h, w)
    out = cv2.remap(img, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.reshape((h * w,) + img.shape[2:])[:n]


def reflectance(dn, u, v, cam, max_radius_px=1400):
    """DJI Mavic 3M image processing guide (relative reflectance; scale cancels in indices)."""
    m = cam.meta
    k = np.array(list(map(float, m['VignettingData'].split(','))))
    r = np.hypot(u - cam.ocx, v - cam.ocy)
    V = 1.0 + sum(k[i] * r ** (i + 1) for i in range(6))
    x = (dn.astype(np.float64) - float(m['BlackLevel'])) / 65536.0 * V / (float(m['SensorGain']) * float(m['ExposureTime']) * 1e-6)
    return x * float(m['SensorGainAdjustment']) / float(m['Irradiance']), (dn >= 65000) | (dn <= 0) | (r > max_radius_px)


def read_band(path):
    import tifffile
    return tifffile.imread(path).astype(np.float32)


def read_display(path):
    """8-bit BGR image for display (JPG as is, TIF stretched)."""
    if path.upper().endswith(('.JPG', '.JPEG')):
        return cv2.imread(path)
    t = read_band(path); v = t[t > 0]
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    g = np.clip((t - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


# ------------------------------------------------------------------ batches --
PAT = re.compile(r'^(.*)_(D\.JPG|MS_(G|R|RE|NIR)\.TIF)$', re.I)


def discover(folder):
    caps = defaultdict(dict)
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if not d.startswith('.')]     # .Trashes, .Spotlight-V100, ...
        for f in files:
            if f.startswith('.'):        # macOS '._xxx' AppleDouble files on exFAT/FAT drives, .DS_Store
                continue
            mt = PAT.match(f)
            if mt:
                band = 'RGB' if mt.group(2).upper().startswith('D') else mt.group(3).upper()
                caps[mt.group(1)][band] = os.path.join(root, f)
    return dict(sorted(caps.items()))


def resolve_image(arg, folder, caps):
    """Accept a full path, a file name inside the folder, or a capture name / number."""
    if os.path.isfile(arg):
        return os.path.abspath(arg)
    name = os.path.basename(arg).upper()
    for key, files in caps.items():
        for band, p in files.items():
            if os.path.basename(p).upper() == name:
                return p
    hits = [k for k in caps if name in k.upper()]
    if len(hits) == 1:
        files = caps[hits[0]]
        return files.get('RGB') or files.get('NIR')
    sys.exit(f'could not find image "{arg}" in {folder}' + (f' ({len(hits)} captures match)' if hits else ''))


# ------------------------------------------------------------------- picker --
def pick_points(paths, titles, previous=None):
    import matplotlib.pyplot as plt
    imgs = [cv2.cvtColor(read_display(p), cv2.COLOR_BGR2RGB) for p in paths]
    fig, axes = plt.subplots(1, 2, figsize=(17, 8.5))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03, wspace=0.04)
    for ax, im, t in zip(axes, imgs, titles):
        ax.imshow(im); ax.set_title(t, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Click the SAME ground-level object in both images (post base, stone, pipe joint, bed end).\n'
                 'Scroll = zoom   right-drag = pan   click again = move mark   r = reset view   '
                 'Enter = done   Esc = cancel', fontsize=11)
    pts = [None, None] if previous is None else [tuple(p) for p in previous]
    marks = [[], []]; status = {'cancel': False}; drag = {}
    home = [(ax.get_xlim(), ax.get_ylim()) for ax in axes]

    def draw_mark(k):
        for a in marks[k]:
            a.remove()
        marks[k] = []
        if pts[k] is None:
            return
        x, y = pts[k]; ax = axes[k]
        marks[k] += ax.plot(x, y, marker='+', color='red', ms=28, mew=1.5)
        marks[k] += ax.plot(x, y, marker='o', mfc='none', mec='yellow', ms=14, mew=1.5)
        axes[k].set_xlabel(f'u = {x:.1f}   v = {y:.1f}', fontsize=10)

    for k in range(2):
        draw_mark(k)

    def busy():
        tb = getattr(fig.canvas, 'toolbar', None)
        return tb is not None and bool(getattr(tb, 'mode', ''))

    def on_press(ev):
        if ev.inaxes not in axes or busy():
            return
        k = list(axes).index(ev.inaxes)
        if ev.button == 1 and ev.xdata is not None:
            pts[k] = (float(ev.xdata), float(ev.ydata)); draw_mark(k); fig.canvas.draw_idle()
        elif ev.button == 3:
            drag.update(ax=ev.inaxes, x=ev.x, y=ev.y, xl=ev.inaxes.get_xlim(), yl=ev.inaxes.get_ylim())

    def on_motion(ev):
        if not drag or ev.x is None:
            return
        ax = drag['ax']; bb = ax.get_window_extent()
        sx = (drag['xl'][1] - drag['xl'][0]) / bb.width; sy = (drag['yl'][1] - drag['yl'][0]) / bb.height
        dx, dy = (ev.x - drag['x']) * sx, (ev.y - drag['y']) * sy
        ax.set_xlim(drag['xl'][0] - dx, drag['xl'][1] - dx); ax.set_ylim(drag['yl'][0] - dy, drag['yl'][1] - dy)
        fig.canvas.draw_idle()

    def on_release(ev):
        drag.clear()

    def on_scroll(ev):
        ax = ev.inaxes
        if ax not in axes or ev.xdata is None:
            return
        f = 0.75 if ev.button == 'up' else 1 / 0.75
        xl, yl = ax.get_xlim(), ax.get_ylim()
        ax.set_xlim(ev.xdata + (xl[0] - ev.xdata) * f, ev.xdata + (xl[1] - ev.xdata) * f)
        ax.set_ylim(ev.ydata + (yl[0] - ev.ydata) * f, ev.ydata + (yl[1] - ev.ydata) * f)
        fig.canvas.draw_idle()

    def on_key(ev):
        if ev.key == 'enter':
            if None in pts:
                fig.suptitle('Mark the object in BOTH images first, then press Enter.', color='red'); fig.canvas.draw_idle()
                return
            plt.close(fig)
        elif ev.key == 'escape':
            status['cancel'] = True; plt.close(fig)
        elif ev.key == 'r':
            for ax, (xl, yl) in zip(axes, home):
                ax.set_xlim(xl); ax.set_ylim(yl)
            fig.canvas.draw_idle()

    for name, fn in (('button_press_event', on_press), ('motion_notify_event', on_motion),
                     ('button_release_event', on_release), ('scroll_event', on_scroll), ('key_press_event', on_key)):
        fig.canvas.mpl_connect(name, fn)
    plt.show()
    return None if status['cancel'] or None in pts else pts


# --------------------------------------------------------------------- grid --
class Grid:
    """Tiles W x H metres. Tile (0,0) = tile X, centred on the reference object.
    i grows along axis_i (roughly 'right' in the reference image), j along axis_j ('down')."""
    def __init__(self, origin, ax_i, ax_j, W, H):
        self.o = np.asarray(origin, float); self.ui = np.asarray(ax_i, float); self.uj = np.asarray(ax_j, float)
        self.W, self.H = float(W), float(H)

    def to_tile(self, E, N):          # fractional tile coordinates (tile i spans [i, i+1))
        de, dn = np.asarray(E) - self.o[0], np.asarray(N) - self.o[1]
        return (de * self.ui[0] + dn * self.ui[1]) / self.W + 0.5, (de * self.uj[0] + dn * self.uj[1]) / self.H + 0.5

    def to_ground(self, fi, fj):
        s = (np.asarray(fi, float) - 0.5) * self.W; t = (np.asarray(fj, float) - 0.5) * self.H
        return self.o[0] + s * self.ui[0] + t * self.uj[0], self.o[1] + s * self.ui[1] + t * self.uj[1]

    def corners(self, i, j):
        return np.stack(self.to_ground([i, i + 1, i + 1, i], [j, j, j + 1, j + 1]), 1)

    def center(self, i, j):
        return self.to_ground(i + 0.5, j + 0.5)

    def outline(self, i, j, n=10):
        a = np.linspace(0, 1, n)
        fi = np.r_[i + a, np.full(n, i + 1), i + 1 - a, np.full(n, i)]
        fj = np.r_[np.full(n, j), j + a, np.full(n, j + 1), j + 1 - a]
        return self.to_ground(fi, fj)

    def tiles_in(self, poly):
        fi, fj = self.to_tile(poly[:, 0], poly[:, 1])
        ii, jj = np.meshgrid(np.arange(int(np.floor(fi.min())), int(np.floor(fi.max())) + 1),
                             np.arange(int(np.floor(fj.min())), int(np.floor(fj.max())) + 1), indexing='ij')
        ii, jj = ii.ravel(), jj.ravel()
        full = np.ones(ii.size, bool); anyin = np.zeros(ii.size, bool)
        for di, dj in [(0, 0), (1, 0), (1, 1), (0, 1), (.5, 0), (1, .5), (.5, 1), (0, .5), (.5, .5)]:
            ins = points_in_poly(*self.to_ground(ii + di, jj + dj), poly)
            full &= ins; anyin |= ins
        return ii, jj, full, anyin


def tid(i, j):
    return f'{i:+d},{j:+d}'


def detect_rows(cam, img, center):
    """Row direction (unit ground vector) and coherence from a 12 x 12 m patch around `center`."""
    st = 0.01; n = 1200
    ee, nn = np.meshgrid(center[0] - 6 + st * np.arange(n), center[1] + 6 - st * np.arange(n))
    u, v = cam.ground_to_pixel(ee, nn)
    o = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)
    g = cv2.cvtColor(o, cv2.COLOR_BGR2GRAY).astype(np.float32) if o.ndim == 3 else o.astype(np.float32)
    valid = cv2.erode((g > 0).astype(np.uint8), np.ones((25, 25), np.uint8)) > 0
    g = cv2.GaussianBlur(g, (0, 0), 8)
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0)[valid]; gy = cv2.Sobel(g, cv2.CV_64F, 0, 1)[valid]
    jxx, jyy, jxy = (gx * gx).sum(), (gy * gy).sum(), (gx * gy).sum()
    th = 0.5 * math.atan2(2 * jxy, jxx - jyy)
    coh = math.hypot(jxx - jyy, 2 * jxy) / (jxx + jyy + 1e-12)
    gdir = np.array([math.cos(th), -math.sin(th)])            # gradient (across rows), ground E,N
    return np.array([-gdir[1], gdir[0]]), coh


def make_grid(mode, ref_cam, ref_img, P, W, H):
    right, down = ref_cam.right, -ref_cam.fwd
    info = {}
    if mode == 'image':
        a = None
    elif mode == 'rows':
        a, coh = detect_rows(ref_cam, ref_img, P)
        if coh < 0.4:
            a2, coh2 = detect_rows(ref_cam, ref_img, (ref_cam.ce, ref_cam.cn))
            print(f'  no clear rows around the reference object (coherence {coh:.2f}); '
                  f'whole reference frame: coherence {coh2:.2f}')
            if coh2 > coh:
                a, coh = a2, coh2
        info['row_coherence'] = round(coh, 3)
        if coh < 0.4:
            print('  rows not clear in the reference frame either; aligning tiles with the image axes '
                  '(use --angle <row azimuth> to force a direction)')
            a = None
    else:
        az = math.radians(float(mode)); a = np.array([math.sin(az), math.cos(az)])
    if a is None:
        ui, uj = right, down
    else:
        c = np.array([a[1], -a[0]])
        if abs(a @ right) >= abs(a @ down):
            ui, uj = a * np.sign(a @ right), c * np.sign(c @ down)
        else:
            ui, uj = c * np.sign(c @ right), a * np.sign(a @ down)
    az_i = math.degrees(math.atan2(ui[0], ui[1])) % 360; az_j = math.degrees(math.atan2(uj[0], uj[1])) % 360
    info.update(axis_i_azimuth_deg=round(az_i, 2), axis_j_azimuth_deg=round(az_j, 2))
    if a is not None:
        info['row_azimuth_deg'] = round(math.degrees(math.atan2(a[0], a[1])) % 180, 2)
    return Grid(P, ui, uj, W, H), info


# --------------------------------------------------------------- drawing ----
def draw_tiles(img, cam, grid, tiles, sc, highlight=(0, 0), labels=True):
    """tiles: list of (i, j, fully_inside)."""
    for i, j, full in sorted(tiles, key=lambda t: (t[0], t[1]) == highlight):
        E, N = grid.outline(i, j)
        u, v = cam.ground_to_pixel(E, N)
        if (u <= OUTSIDE / 2).any():
            continue
        pts = (np.c_[u, v] * sc).round().astype(np.int32)
        hx = (i, j) == highlight
        col = (0, 0, 255) if hx else ((0, 255, 255) if full else (170, 170, 170))
        th = max(2, int(round(5 * sc * 2.5))) if hx else (max(1, int(round(2 * sc * 2.5))) if full else 1)
        cv2.polylines(img, [pts], True, col, th, cv2.LINE_AA)
        if labels and (full or hx):
            E, N = grid.center(i, j); u, v = cam.ground_to_pixel(np.array([E]), np.array([N]))
            w = max(10, min(np.ptp(pts[:, 0]), np.ptp(pts[:, 1])))
            txt = 'X' if hx else tid(i, j)
            fs = float(np.clip(w / (22 * len(txt) + 10), 0.3, 2.5)); t2 = max(1, int(round(fs * 1.8)))
            (tw, tht), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, t2)
            p = (int(u[0] * sc - tw / 2), int(v[0] * sc + tht / 2))
            cv2.putText(img, txt, p, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), t2 + 2, cv2.LINE_AA)
            cv2.putText(img, txt, p, cv2.FONT_HERSHEY_SIMPLEX, fs, col, t2, cv2.LINE_AA)


def mark_point(img, uv, sc):
    x, y = int(uv[0] * sc), int(uv[1] * sc); r = max(8, int(40 * sc))
    cv2.circle(img, (x, y), r, (255, 0, 255), max(2, int(4 * sc)), cv2.LINE_AA)
    cv2.line(img, (x - 2 * r, y), (x + 2 * r, y), (255, 0, 255), max(1, int(3 * sc)), cv2.LINE_AA)
    cv2.line(img, (x, y - 2 * r), (x, y + 2 * r), (255, 0, 255), max(1, int(3 * sc)), cv2.LINE_AA)


def label_bar(img, text, h=34):
    bar = np.full((h, img.shape[1], 3), 25, np.uint8)
    fs = h / 50
    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0]
    if tw > img.shape[1] - 12:
        fs *= (img.shape[1] - 12) / tw
    cv2.putText(bar, text, (6, int(h * 0.68)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def auto_level(img):
    v = img[img.sum(2) > 0]
    if v.size == 0:
        return img
    lo, hi = np.percentile(v, 1), np.percentile(v, 99.5)
    return np.clip((img.astype(np.float32) - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ref_folder', help='reference batch folder (e.g. the earlier flight)')
    ap.add_argument('other_folder', help='batch to compare against the reference')
    ap.add_argument('--ref-image', required=True, help='image of the reference batch showing the object (path, file name or capture number)')
    ap.add_argument('--other-image', required=True, help='image of the other batch showing the same object')
    ap.add_argument('--tile', nargs=2, type=float, default=[1.25, 2.5], metavar=('W', 'H'),
                    help='tile size in metres: W along axis i (~right in the ref image), H along axis j (~down). default 1.25 2.5')
    ap.add_argument('--angle', default='rows', help="tile orientation: 'rows' (auto, follow crop rows), 'image' (axes of the reference image) or a row azimuth in degrees")
    ap.add_argument('--out', default='tiles_out')
    ap.add_argument('--repick', action='store_true', help='open the picker even if reference.json exists')
    ap.add_argument('--ref-point', nargs=2, type=float, metavar=('U', 'V'), help='skip the picker: pixel in --ref-image')
    ap.add_argument('--other-point', nargs=2, type=float, metavar=('U', 'V'), help='skip the picker: pixel in --other-image')
    ap.add_argument('--overlay-scale', type=float, default=0.4)
    ap.add_argument('--no-overlays', action='store_true')
    ap.add_argument('--no-stats', action='store_true', help='only place tiles (skip NDVI etc.)')
    ap.add_argument('--chips', type=int, default=24)
    ap.add_argument('--rgb-pp-shift', nargs=2, type=float, default=[-50, -20], help=argparse.SUPPRESS)
    args = ap.parse_args()
    t0 = time.time()
    out = args.out; os.makedirs(out, exist_ok=True)

    batches = {'ref': os.path.abspath(args.ref_folder), 'other': os.path.abspath(args.other_folder)}
    names = {k: os.path.basename(os.path.normpath(v)) or k for k, v in batches.items()}
    if names['ref'] == names['other']:
        names = {'ref': 'ref', 'other': 'other'}
    caps = {k: discover(v) for k, v in batches.items()}
    for k in caps:
        print(f'{names[k]}: {len(caps[k])} captures in {batches[k]}')
        if not caps[k]:
            sys.exit('no DJI *_D.JPG / *_MS_*.TIF files found')
    img_path = {'ref': resolve_image(args.ref_image, batches['ref'], caps['ref']),
                'other': resolve_image(args.other_image, batches['other'], caps['other'])}

    # ---- 1. reference clicks
    ref_json = os.path.join(out, 'reference.json')
    clicks = None
    if args.ref_point and args.other_point:
        clicks = [tuple(args.ref_point), tuple(args.other_point)]
    elif os.path.exists(ref_json) and not args.repick:
        old = json.load(open(ref_json))
        if (os.path.basename(old['ref_image']) == os.path.basename(img_path['ref'])
                and os.path.basename(old['other_image']) == os.path.basename(img_path['other'])):
            clicks = [tuple(old['ref_pixel']), tuple(old['other_pixel'])]
            print(f'using the reference clicks saved in {ref_json} (add --repick to click again)')
    if clicks is None:
        prev = None
        if os.path.exists(ref_json):
            old = json.load(open(ref_json))
            if os.path.basename(old['ref_image']) == os.path.basename(img_path['ref']):
                prev = [old['ref_pixel'], old['other_pixel']]
        print('opening the picker ...')
        clicks = pick_points([img_path['ref'], img_path['other']],
                             [f"{names['ref']}: {os.path.basename(img_path['ref'])}",
                              f"{names['other']}: {os.path.basename(img_path['other'])}"], prev)
        if clicks is None:
            sys.exit('cancelled - no reference picked')

    # ---- 2. offset + grid
    m0 = read_xmp(img_path['ref'])
    frame = LocalFrame(float(m0['GpsLatitude']), float(m0['GpsLongitude']))
    pp = tuple(args.rgb_pp_shift)
    ref_cam = Cam(img_path['ref'], frame, rgb_pp_shift=pp)
    oth_cam_raw = Cam(img_path['other'], frame, rgb_pp_shift=pp)
    P = np.array(ref_cam.pixel_to_ground(clicks[0][0], clicks[0][1]), float)
    Q = np.array(oth_cam_raw.pixel_to_ground(clicks[1][0], clicks[1][1]), float)
    offset = P - Q                                         # applied to every frame of the other batch
    offsets = {'ref': (0.0, 0.0), 'other': (float(offset[0]), float(offset[1]))}
    ref_img = read_display(img_path['ref'])
    grid, ginfo = make_grid(args.angle, ref_cam, ref_img, P, args.tile[0], args.tile[1])
    latP, lonP = frame.to_ll(*P); latQ, lonQ = frame.to_ll(*Q)
    ref = {
        'ref_folder': batches['ref'], 'other_folder': batches['other'],
        'ref_image': img_path['ref'], 'ref_pixel': [round(clicks[0][0], 2), round(clicks[0][1], 2)],
        'other_image': img_path['other'], 'other_pixel': [round(clicks[1][0], 2), round(clicks[1][1], 2)],
        'object_latlon_from_ref_batch': [round(latP, 9), round(lonP, 9)],
        'object_latlon_from_other_batch_metadata': [round(latQ, 9), round(lonQ, 9)],
        'other_batch_offset_m': {'dE': round(offsets['other'][0], 4), 'dN': round(offsets['other'][1], 4),
                                 'distance': round(float(np.hypot(*offset)), 4)},
        'grid': dict(tile_X_center_latlon=[round(latP, 9), round(lonP, 9)], tile_w_m=args.tile[0], tile_h_m=args.tile[1],
                     angle_mode=args.angle, **ginfo),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    json.dump(ref, open(ref_json, 'w'), indent=2)
    print(f"reference object: {latP:.8f}, {lonP:.8f}")
    print(f"{names['other']} batch offset: dE={offset[0]:+.3f} m  dN={offset[1]:+.3f} m  (|d| = {np.hypot(*offset):.3f} m)  -> saved in {ref_json}")
    print(f"tiles {args.tile[0]} x {args.tile[1]} m, axis i azimuth {ginfo['axis_i_azimuth_deg']} deg, axis j {ginfo['axis_j_azimuth_deg']} deg"
          + (f", rows at {ginfo['row_azimuth_deg']} deg (coherence {ginfo['row_coherence']})" if 'row_azimuth_deg' in ginfo else ''))

    # ---- 3. place tiles in every frame
    cams = {}
    def cam(b, key, kind):
        k = (b, key, kind)
        if k not in cams:
            cams[k] = Cam(caps[b][key][kind], frame, offsets[b], pp)
        return cams[k]

    # drop files whose DJI metadata cannot be read (corrupt / partial copies) with a warning
    for b in batches:
        for key in list(caps[b]):
            for kind in list(caps[b][key]):
                try:
                    cam(b, key, kind)
                except Exception as e:
                    print(f'  skipping {os.path.basename(caps[b][key][kind])}: {e}')
                    del caps[b][key][kind]
            if not any(k in caps[b][key] for k in ('RGB', 'NIR')):
                del caps[b][key]

    visible = {b: {} for b in batches}           # RGB (or NIR) tiles per frame, for drawing
    owner = {b: {} for b in batches}             # tile -> (capture, distance to frame centre) using NIR footprint
    place_rows = []
    for b in batches:
        for key, files in caps[b].items():
            for kind in FILE_KINDS:
                if kind not in files:
                    continue
                c = cam(b, key, kind)
                ii, jj, full, anyin = grid.tiles_in(c.footprint())
                if kind == ('RGB' if 'RGB' in files else 'NIR'):
                    visible[b][key] = [(int(i), int(j), bool(f)) for i, j, f, a in zip(ii, jj, full, anyin) if a]
                if kind == 'NIR':
                    for i, j in zip(ii[full], jj[full]):
                        E, N = grid.center(i, j); d = math.hypot(E - c.ce, N - c.cn)
                        if (int(i), int(j)) not in owner[b] or d < owner[b][(int(i), int(j))][1]:
                            owner[b][(int(i), int(j))] = (key, d)
                for i, j in zip(ii[full], jj[full]):
                    cs = grid.corners(i, j); u, v = c.ground_to_pixel(cs[:, 0], cs[:, 1])
                    E, N = grid.center(i, j); uc, vc = c.ground_to_pixel(np.array([E]), np.array([N]))
                    place_rows.append([names[b], key, os.path.basename(c.path), tid(i, j), int(i), int(j),
                                       f'{uc[0]:.1f}', f'{vc[0]:.1f}'] + [f'{x:.1f}' for p in zip(u, v) for x in p])
        print(f'{names[b]}: tiles placed in {len(caps[b])} frames')
    with open(os.path.join(out, 'tile_placements.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['batch', 'capture', 'file', 'tile', 'i', 'j', 'center_u', 'center_v',
                    'u1', 'v1', 'u2', 'v2', 'u3', 'v3', 'u4', 'v4'])
        w.writerows(place_rows)
    print(f'wrote tile_placements.csv ({len(place_rows)} tile placements)')

    # ---- 4. reference check: tile X in every frame that sees it
    reference_check(caps, batches, names, cam, grid, img_path, clicks, out)

    # ---- 5. overlays
    if not args.no_overlays:
        sc = args.overlay_scale
        for b in batches:
            od = os.path.join(out, 'overlays', names[b]); os.makedirs(od, exist_ok=True)
            for key, tl in visible[b].items():
                kind = 'RGB' if 'RGB' in caps[b][key] else 'NIR'
                c = cam(b, key, kind)
                img = cv2.resize(read_display(c.path), None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                draw_tiles(img, c, grid, tl, sc)
                if os.path.abspath(img_path[b]) in [os.path.abspath(p) for p in caps[b][key].values()]:
                    u, v = c.ground_to_pixel(np.array([P[0]]), np.array([P[1]])); mark_point(img, (u[0], v[0]), sc)
                cv2.imwrite(os.path.join(od, f'{key}_tiles.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print('wrote overlays/')

    # ---- 6. statistics
    if not args.no_stats:
        stats = tile_stats(caps, batches, names, cam, grid, owner)
        write_tables(stats, owner, grid, frame, names, out)
        if args.chips:
            make_chips(caps, cam, grid, owner, stats, names, out, args.chips)
    print(f'done in {time.time() - t0:.0f} s -> {os.path.abspath(out)}')


def reference_check(caps, batches, names, cam, grid, img_path, clicks, out, max_frames=12):
    E0, N0 = grid.center(0, 0)
    panels = []
    for b in batches:
        found = []
        for key, files in caps[b].items():
            kind = 'RGB' if 'RGB' in files else 'NIR'
            c = cam(b, key, kind)
            if points_in_poly(np.array([E0]), np.array([N0]), c.footprint(inset_px=0))[0]:
                clicked = os.path.abspath(img_path[b]) in [os.path.abspath(p) for p in files.values()]
                found.append((not clicked, math.hypot(E0 - c.ce, N0 - c.cn), key, c))
        found.sort(key=lambda x: (x[0], x[1]))
        for is_other, d, key, c in found[:max_frames]:
            img = read_display(c.path)
            ii, jj, full, anyin = grid.tiles_in(c.footprint(inset_px=0))
            tl = [(int(i), int(j), bool(f)) for i, j, f, a in zip(ii, jj, full, anyin) if a and abs(i) <= 2 and abs(j) <= 2]
            if not tl:
                continue
            draw_tiles(img, c, grid, tl, 1.0)
            if not is_other:
                u, v = c.ground_to_pixel(np.array([E0]), np.array([N0])); mark_point(img, (u[0], v[0]), 1.0)
            cs = np.concatenate([grid.corners(i, j) for i, j, f in tl])
            u, v = c.ground_to_pixel(cs[:, 0], cs[:, 1])
            x0, x1 = int(max(0, u.min() - 40)), int(min(c.size[0], u.max() + 40))
            y0, y1 = int(max(0, v.min() - 40)), int(min(c.size[1], v.max() + 40))
            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            s = 520 / crop.shape[0]
            crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), 520), interpolation=cv2.INTER_AREA)
            tag = 'CLICKED IMAGE' if not is_other else f'{d:.1f} m from frame centre'
            if crop.shape[1] < 420:
                crop = cv2.copyMakeBorder(crop, 0, 0, 0, 420 - crop.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
            panels.append(label_bar(label_bar(crop, key, 26), f'{names[b]}  -  {tag}', 30))
        print(f'{names[b]}: tile X visible in {len(found)} frames')
    if panels:
        rows, row, wsum = [], [], 0
        for p in panels:
            if row and wsum + p.shape[1] > 2400:
                rows.append(row); row, wsum = [], 0
            row.append(p); wsum += p.shape[1] + 8
        rows.append(row)
        width = max(sum(p.shape[1] + 8 for p in r) for r in rows)
        sheet = []
        for r in rows:
            line = np.hstack(sum([[p, np.full((p.shape[0], 8, 3), 255, np.uint8)] for p in r], []))
            sheet.append(cv2.copyMakeBorder(line, 0, 8, 0, width - line.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)))
        cv2.imwrite(os.path.join(out, 'reference_check.jpg'), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print('wrote reference_check.jpg (tile X in red; magenta cross = your reference object)')


def tile_stats(caps, batches, names, cam, grid, owner, step=0.02, thr=0.5):
    nw, nh = max(2, int(round(grid.W / step))), max(2, int(round(grid.H / step)))
    LI, LJ = np.meshgrid((np.arange(nw) + 0.5) / nw, (np.arange(nh) + 0.5) / nh, indexing='ij')
    LI, LJ = LI.ravel(), LJ.ravel(); per = LI.size
    stats = {b: {} for b in batches}
    for b in batches:
        bykey = defaultdict(list)
        for (i, j), (key, d) in owner[b].items():
            bykey[key].append((i, j, d))
        for n_, (key, tl) in enumerate(sorted(bykey.items())):
            fi = np.concatenate([i + LI for i, j, d in tl]); fj = np.concatenate([j + LJ for i, j, d in tl])
            E, N = grid.to_ground(fi, fj)
            refl = {}; bad = np.zeros(E.size, bool)
            for band in BANDS:
                if band in caps[b][key]:
                    c = cam(b, key, band); u, v = c.ground_to_pixel(E, N)
                    r_, b_ = reflectance(remap_points(read_band(c.path), u, v), u, v, c)
                    refl[band] = r_; bad |= b_
            if 'NIR' not in refl or 'R' not in refl:
                continue
            with np.errstate(divide='ignore', invalid='ignore'):
                ndvi = (refl['NIR'] - refl['R']) / (refl['NIR'] + refl['R'])
                ndre = (refl['NIR'] - refl['RE']) / (refl['NIR'] + refl['RE']) if 'RE' in refl else np.full_like(ndvi, np.nan)
            bad |= ~np.isfinite(ndvi)
            for k, (i, j, d) in enumerate(tl):
                sl = slice(k * per, (k + 1) * per); ok = ~bad[sl]
                rec = {'capture': key, 'dist_from_frame_centre_m': d, 'valid_frac': ok.mean()}
                if ok.sum() > 10:
                    nd = ndvi[sl][ok]; veg = nd > thr
                    rec.update(ndvi=nd.mean(), ndvi_plants=nd[veg].mean() if veg.any() else np.nan,
                               ndre=np.nanmean(ndre[sl][ok]), plant_cover=veg.mean())
                stats[b][(i, j)] = rec
            print(f'\r  {names[b]}: measured tiles in {n_ + 1}/{len(bykey)} frames', end='')
        print()
    return stats


def write_tables(stats, owner, grid, frame, names, out):
    keys = ['capture', 'dist_from_frame_centre_m', 'valid_frac', 'ndvi', 'ndvi_plants', 'ndre', 'plant_cover']
    R, O = names['ref'], names['other']
    allt = sorted(set(stats['ref']) | set(stats['other']), key=lambda t: (t[1], t[0]))
    cols = ['tile', 'i', 'j', 'is_tile_X', 'center_lat', 'center_lon']
    cols += [f'{R}_{k}' for k in keys] + [f'{O}_{k}' for k in keys]
    cols += [f'change_{k}' for k in ('ndvi', 'ndvi_plants', 'ndre', 'plant_cover')]
    fmt = lambda v: '' if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f'{v:.4f}' if isinstance(v, (float, np.floating)) else v)
    rows, feats = [], []
    for (i, j) in allt:
        E, N = grid.center(i, j); lat, lon = frame.to_ll(E, N)
        r = {'tile': tid(i, j), 'i': i, 'j': j, 'is_tile_X': int((i, j) == (0, 0)), 'center_lat': f'{lat:.8f}', 'center_lon': f'{lon:.8f}'}
        for b, nm in (('ref', R), ('other', O)):
            s = stats[b].get((i, j), {})
            for k in keys:
                r[f'{nm}_{k}'] = fmt(s.get(k))
        for k in ('ndvi', 'ndvi_plants', 'ndre', 'plant_cover'):
            a, c = stats['other'].get((i, j), {}).get(k), stats['ref'].get((i, j), {}).get(k)
            r[f'change_{k}'] = fmt(a - c) if a is not None and c is not None and np.isfinite(a) and np.isfinite(c) else ''
        rows.append(r)
        ring = [[round(x, 9) for x in frame.to_ll(e, n)[::-1]] for e, n in grid.corners(i, j)]; ring.append(ring[0])
        feats.append({'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [ring]}, 'properties': r})
    with open(os.path.join(out, 'tiles.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    json.dump({'type': 'FeatureCollection', 'features': feats}, open(os.path.join(out, 'tiles.geojson'), 'w'))
    nboth = sum(1 for r in rows if r['change_ndvi'] != '')
    print(f'wrote tiles.csv / tiles.geojson: {len(rows)} tiles, {nboth} measured in both batches')


def make_chips(caps, cam, grid, owner, stats, names, out, n):
    both = sorted(t for t in owner['ref'] if t in owner['other'] and 'ndvi' in stats['ref'].get(t, {}) and 'ndvi' in stats['other'].get(t, {}))
    if not both:
        return
    sel = sorted(set(random.Random(0).sample(both, min(n - 1, len(both))) + ([(0, 0)] if (0, 0) in both else [])), key=lambda t: (t != (0, 0), t[1], t[0]))
    od = os.path.join(out, 'chips'); os.makedirs(od, exist_ok=True)
    m, st = 0.3, 0.01; cache = {}; rows = []
    for (i, j) in sel:
        pan = []
        for b in ('ref', 'other'):
            key = owner[b][(i, j)][0]; kind = 'RGB' if 'RGB' in caps[b][key] else 'NIR'
            if (b, key) not in cache:
                if len(cache) > 3:
                    cache.clear()
                cache[(b, key)] = read_display(cam(b, key, kind).path)
            c = cam(b, key, kind)
            s = np.arange(-m, grid.W + m, st) + st / 2; t = np.arange(-m, grid.H + m, st) + st / 2
            S, T = np.meshgrid(s / grid.W + i, t / grid.H + j)
            E, N = grid.to_ground(S, T); u, v = c.ground_to_pixel(E, N)
            chip = auto_level(cv2.remap(cache[(b, key)], u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR))
            x0 = int(round(m / st))
            cv2.rectangle(chip, (x0, x0), (chip.shape[1] - x0, chip.shape[0] - x0), (0, 0, 255) if (i, j) == (0, 0) else (0, 255, 255), 2)
            sr = stats[b][(i, j)]
            pan.append(label_bar(chip, f"{names[b]}  NDVI {sr['ndvi']:.2f}  cover {sr['plant_cover']:.2f}", 26))
        row = np.hstack([pan[0], np.full((pan[0].shape[0], 6, 3), 255, np.uint8), pan[1]])
        rows.append(label_bar(row, f"tile {'X' if (i, j) == (0, 0) else tid(i, j)}", 26))
    for k in range(0, len(rows), 6):
        sheet = [np.vstack([r, np.full((8, r.shape[1], 3), 255, np.uint8)]) for r in rows[k:k + 6]]
        wmax = max(s.shape[1] for s in sheet)
        sheet = [cv2.copyMakeBorder(s, 0, 0, 0, wmax - s.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for s in sheet]
        cv2.imwrite(os.path.join(od, f'chips_{k // 6 + 1:02d}.jpg'), np.vstack(sheet), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f'wrote chips/ ({len(rows)} tiles; each chip is oriented like the reference image)')


if __name__ == '__main__':
    main()
