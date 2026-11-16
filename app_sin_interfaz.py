"""SxMOV: SPHEREx Moving Object Visualizator.
Shows objects with relative motion with respect to background stars.
Run locally:  streamlit run app.py
"""
import io, re, time, warnings, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from urllib.parse import quote_plus

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.time import Time
from astropy.utils.data import conf
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astroquery.ipac.irsa import Irsa
from astroquery.simbad import Simbad
from PIL import Image, ImageDraw
from reproject import reproject_interp

logging.getLogger("astropy").setLevel(logging.ERROR)
conf.remote_timeout = 120
Irsa.ROW_LIMIT = 10000
warnings.filterwarnings("ignore")

PIX = 6.15  # SPHEREx pixel scale [arcsec/pixel]
BANDS = {1: (0.75, 1.11), 2: (1.11, 1.64), 3: (1.64, 2.42),
         4: (2.42, 3.82), 5: (3.82, 4.42), 6: (4.42, 5.00)}
ZOOMS = [1, 1.5, 2, 3, 4, 6, 8]
OUT = 512
COORD_DECIMALS = 4  # coordinates equal up to 4 decimals (~0.36") count as the same field

st.set_page_config(page_title="SxMOV: SPHEREx Moving Object Visualizator", layout="wide")
st.title("SxMOV: SPHEREx Moving Object Visualizator")


# ================================================================ data access
@st.cache_data(ttl=3600)
def list_releases():
    try:
        t = Irsa.list_collections(servicetype="SIA", filter="spherex")
        names = sorted(str(c) for c in t["collection"])
    except Exception:
        names = ["spherex_dr1", "spherex_dr1_deep", "spherex_qr3", "spherex_qr3_deep"]
    return [n for n in names if re.fullmatch(r"spherex_(dr|qr)\d+(_deep)?", n)]


def latest(names):
    best = {}
    for n in names:
        m = re.fullmatch(r"spherex_(dr|qr)(\d+)", n)
        if m and (m[1] not in best or int(m[2]) > best[m[1]][0]):
            best[m[1]] = (int(m[2]), n)
    return [v[1] for v in best.values()]


def _simbad_astroquery(name):
    """SIMBAD through astroquery. The hard limit is set by hand to avoid the /capabilities 404."""
    last = None
    for server in ("simbad.cds.unistra.fr", "simbad.cfa.harvard.edu"):
        try:
            sim = Simbad()
            try:
                sim.server = server
            except Exception:
                pass
            sim._hardlimit = 2_000_000
            for f in ("otype", "sp", "parallax", "propermotions"):
                try:
                    sim.add_votable_fields(f)
                except Exception:
                    pass
            t = sim.query_object(name)
            if t is None or len(t) == 0:
                return None
            row = t[0]
            info = {c: str(row[c]) for c in t.colnames if str(row[c]) not in ("--", "")}
            desc = ""
            try:
                d = sim.query_tap(f"SELECT description FROM otypedef WHERE otype = '{str(row['otype'])}'")
                desc = str(d[0][0])
            except Exception:
                pass
            return dict(ra=float(row["ra"]), dec=float(row["dec"]), info=info, desc=desc)
        except Exception as e:
            last = e
    raise last


def _simbad_sesame(name):
    """Plan B: Sesame resolver (CDS) for RA/Dec, then try to complete the info with a direct TAP query."""
    c = SkyCoord.from_name(name)
    ra, dec = float(c.icrs.ra.deg), float(c.icrs.dec.deg)
    info, desc = {}, ""
    try:
        import pyvo
        adql = ("SELECT TOP 1 main_id, otype, sp_type, plx_value, pmra, pmdec FROM basic "
                f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{2/3600}))")
        for server in ("simbad.cds.unistra.fr", "simbad.cfa.harvard.edu"):
            try:
                r = pyvo.dal.TAPService(f"https://{server}/simbad/sim-tap").run_sync(adql).to_table()
                if len(r):
                    info = {k: str(r[0][k]) for k in r.colnames if str(r[0][k]) not in ("--", "")}
                    break
            except Exception:
                continue
    except Exception:
        pass
    info.setdefault("main_id", name)
    return dict(ra=ra, dec=dec, info=info, desc=desc)


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_name(name):
    try:
        return _simbad_astroquery(name)
    except Exception:
        pass
    try:
        return _simbad_sesame(name)
    except Exception:
        return None


def simbad_url(ra, dec, radius_arcmin):
    c = SkyCoord(ra * u.deg, dec * u.deg)
    s = c.to_string("hmsdms", sep=" ", precision=6, alwayssign=True)
    return ("https://simbad.cfa.harvard.edu/simbad/sim-coo?Coord=%09" + quote_plus(s) +
            "&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none"
            f"&Radius={radius_arcmin:.2f}&Radius.unit=arcmin&submit=submit+query&CoordList=")


@st.cache_data(ttl=600, show_spinner=False)
def sia_search(ra, dec, collections):
    res = Irsa.query_sia(pos=(SkyCoord(ra * u.deg, dec * u.deg), 1 * u.arcsec), collection=list(collections))
    t = res.to_table() if hasattr(res, "to_table") else res
    df = pd.DataFrame({"access_url": [v.decode() if isinstance(v, bytes) else str(v) for v in t["access_url"]]})
    for c in ("t_min", "em_min", "em_max"):
        df[c] = np.ma.filled(t[c].astype(float), np.nan) if c in t.colnames else np.nan
    df["bandpass"] = [str(v) for v in t["energy_bandpassname"]] if "energy_bandpassname" in t.colnames else ""
    df["det"] = [int(m[1]) if (m := re.search(r"(\d)\s*$", b)) else 0 for b in df["bandpass"]]
    mid = (df.em_min + df.em_max) / 2
    mid = np.where(mid < 1e-3, mid * 1e6, mid)
    for i in np.where(df.det == 0)[0]:  # unknown detector -> infer from wavelength
        for k, (lo, hi) in BANDS.items():
            if lo <= mid[i] <= hi:
                df.loc[i, "det"] = k
    df = df.dropna(subset=["t_min"]).sort_values("t_min").reset_index(drop=True)
    df["date"] = Time(df.t_min.values, format="mjd").to_datetime()
    return df


def download(uri, ra, dec, size_deg):
    url = f"{uri}{'&' if '?' in uri else '?'}center={ra},{dec}d&size={size_deg}"
    err = None
    for k in range(3):
        try:
            with fits.open(url, cache=False, memmap=False) as hdul:
                hdr = hdul["IMAGE"].header
                data = np.array(hdul["IMAGE"].data, dtype=np.float32)
                zodi = np.array(hdul["ZODI"].data, dtype=np.float32) if "ZODI" in hdul else None
                wcs = WCS(hdr)
                x, y = wcs.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
                wave = np.nan
                try:
                    sw = WCS(hdr, fobj=hdul, key="W"); sw.sip = None
                    wave = float(sw.pixel_to_world(x, y)[0].to(u.um).value)
                except Exception:
                    pass
                meta = {k2: hdr.get(k2) for k2 in ("DETECTOR", "OBSID", "DATE", "PSF_FWHM")}
                meta["VERSION"] = hdul[0].header.get("VERSION", "")
                for key in hdr:  # flagged-pixel counters (L2 N_*)
                    if "N_" in str(key) and isinstance(hdr[key], (int, float)):
                        meta[str(key)] = hdr[key]
                meta["zodi_median"] = float(np.nanmedian(zodi)) if zodi is not None else np.nan
                meta["img_median"] = float(np.nanmedian(data))
                meta["frac_NaN"] = float(np.mean(~np.isfinite(data)))
                scale = float(np.mean(proj_plane_pixel_scales(wcs.celestial)) * 3600)
                return dict(data=data, hdr=hdr.tostring(), x=float(x), y=float(y),
                            wave=wave, meta=meta, scale=scale)
        except Exception as e:
            err = e
            time.sleep(2 * (k + 1))
    return dict(error=str(err))


def aperture_flux(data, x, y, scale, r_ap, r_in, r_out):
    yy, xx = np.indices(data.shape)
    r = np.hypot(xx - x, yy - y) * scale
    ap = (r <= r_ap) & np.isfinite(data)
    ann = (r >= r_in) & (r <= r_out) & np.isfinite(data)
    bkg = np.nanmedian(data[ann]) if ann.sum() > 3 else 0.0
    s = np.nansum(data[ap]) - bkg * ap.sum()
    return s * 1e9 * (scale / 206265.0) ** 2  # MJy/sr -> mJy


GAIA_COLS = ["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"]
VIZIER_TAPS = ("https://tapvizier.cds.unistra.fr/TAPVizieR/tap", "https://vizier.cfa.harvard.edu/TAPVizieR/tap")


def _gaia_irsa(ra, dec, rad, maglim):
    """Gaia DR3 from the IRSA TAP service, if IRSA hosts a Gaia DR3 source table."""
    cats = Irsa.list_catalogs(filter="gaia")
    name = next((k for k, v in cats.items() if "dr3" in (k + v).lower() and "source" in (k + v).lower()), None)
    if name is None:
        raise RuntimeError("no Gaia DR3 table at IRSA")
    q = (f"SELECT TOP 2000 ra, dec, pmra, pmdec, phot_g_mean_mag FROM {name} "
         f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{rad})) "
         f"AND phot_g_mean_mag < {maglim} ORDER BY phot_g_mean_mag")
    return Irsa.query_tap(q).to_table().to_pandas()


def _gaia_vizier_tap(url):
    def fetch(ra, dec, rad, maglim):
        import pyvo
        q = ('SELECT TOP 2000 RA_ICRS, DE_ICRS, pmRA, pmDE, Gmag FROM "I/355/gaiadr3" '
             f"WHERE 1=CONTAINS(POINT('ICRS',RA_ICRS,DE_ICRS),CIRCLE('ICRS',{ra},{dec},{rad})) "
             f"AND Gmag < {maglim} ORDER BY Gmag")
        df = pyvo.dal.TAPService(url).run_sync(q).to_table().to_pandas()
        df.columns = [c.lower() for c in df.columns]
        return df.rename(columns={"ra_icrs": "ra", "de_icrs": "dec", "pmde": "pmdec", "gmag": "phot_g_mean_mag"})
    return fetch


def _gaia_mast(ra, dec, rad, maglim):
    """Gaia DR3 through MAST (archive.stsci.edu)."""
    from astroquery.mast import Catalogs
    try:
        Catalogs.TIMEOUT = 30
    except Exception:
        pass
    t = Catalogs.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=rad * u.deg, catalog="Gaia", version=3)
    df = t.to_pandas(); df.columns = [c.lower() for c in df.columns]
    return df[df.phot_g_mean_mag < maglim].nsmallest(2000, "phot_g_mean_mag")


def _gaia_esa(ra, dec, rad, maglim):
    from astroquery.gaia import Gaia
    q = ("SELECT TOP 2000 ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source "
         f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{rad})) "
         f"AND phot_g_mean_mag < {maglim} ORDER BY phot_g_mean_mag")
    return Gaia.launch_job(q).get_results().to_pandas()


@st.cache_data(ttl=3600, show_spinner=False)
def gaia_stars(ra, dec, rad_deg, maglim):
    """Gaia DR3 cone. All sources are queried in parallel; the first one that answers wins.
    Returns (table, source_name)."""
    sources = [("IRSA", _gaia_irsa), ("VizieR (CDS)", _gaia_vizier_tap(VIZIER_TAPS[0])),
               ("VizieR (CfA)", _gaia_vizier_tap(VIZIER_TAPS[1])), ("MAST", _gaia_mast), ("ESA", _gaia_esa)]
    ex = ThreadPoolExecutor(len(sources))
    futs = {ex.submit(fn, ra, dec, rad_deg, maglim): name for name, fn in sources}
    errors = []
    try:
        for f in as_completed(futs, timeout=60):
            try:
                df = f.result()[GAIA_COLS].copy()
                for c in ("pmra", "pmdec"):
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)  # no proper motion -> 0
                df = df.dropna(subset=["ra", "dec", "phot_g_mean_mag"]).astype(float)
                return df, futs[f]
            except Exception as e:
                errors.append(f"{futs[f]}: {str(e)[:80]}")
    except Exception:
        errors.append("timeout after 60 s")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    raise RuntimeError(" | ".join(errors))


# ================================================================ frames & rendering
def make_frames(items, mode, band, bgsub):
    frames = []
    for g in sorted({i["gid"] for i in items}):
        its = [i for i in items if i["gid"] == g]
        note = ""
        if mode == "band":
            sel = [i for i in its if i["det"] == band]
            if not sel:
                c = np.mean(BANDS[band])
                sel = [min(its, key=lambda i: abs(np.nan_to_num(i["wave"], nan=99) - c))]
                note = "nearest wavelength"
            groups = [sel]
        else:
            groups = [[i for i in its if i["det"] == d] for d in sorted({i["det"] for i in its})]
        arrs = []
        for gr in groups:
            a = np.nanmedian(np.stack([i["rp"] for i in gr]), axis=0)
            arrs.append(a - np.nanmedian(a) if bgsub else a)
        arr = np.nansum(arrs, axis=0) if mode == "sum" else arrs[0]
        mjd = float(np.mean([i["mjd"] for i in its]))
        frames.append(dict(gid=g, mjd=mjd, arr=arr, note=note, n=len(its),
                           label=Time(mjd, format="mjd").strftime("%Y-%m-%d")))
    return frames


def crop_limits(n, zoom):
    half = max(4, int(n / zoom / 2)); c = n // 2
    return max(0, c - half), min(n, c + half)


def gaia_at_epoch(gaia, mjd):
    """Propagate Gaia positions (epoch 2016.0) to the date of the frame using proper motions."""
    dt = 2000 + (mjd - 51544.5) / 365.25 - 2016.0
    ra = gaia.ra + gaia.pmra * dt / 3.6e6 / np.cos(np.deg2rad(gaia.dec))
    dec = gaia.dec + gaia.pmdec * dt / 3.6e6
    return ra.values, dec.values


def sky_to_display(wcs, ra, dec, x0, x1):
    px, py = wcs.world_to_pixel_values(ra, dec)
    w = x1 - x0; s = OUT / w
    return (px - x0 + 0.5) * s, (w - (py - x0) - 0.5) * s


def draw_extras(im, label, mark, width_arcsec):
    d = ImageDraw.Draw(im)
    bar = max([b for b in (10, 20, 30, 60, 120, 300, 600, 1200) if b <= 0.4 * width_arcsec] or [10])
    L = bar / width_arcsec * OUT
    d.line([10, OUT - 14, 10 + L, OUT - 14], fill=(255, 220, 0), width=3)
    d.text((10, OUT - 30), f'{bar}" ({width_arcsec:.0f}" across)', fill=(255, 220, 0))
    if mark:
        d.line([OUT / 2 - 14, OUT / 2, OUT / 2 - 5, OUT / 2], fill=(255, 60, 60), width=2)
        d.line([OUT / 2 + 5, OUT / 2, OUT / 2 + 14, OUT / 2], fill=(255, 60, 60), width=2)
    d.text((8, 8), label, fill=(255, 220, 0))
    return im


def render(frame, lim, stretch, neg, zoom, diff, wcs, gaia, maglim, mark):
    """SPHEREx frame (optionally with Gaia circles overlaid)."""
    a = frame["arr"]; x0, x1 = crop_limits(a.shape[0], zoom)
    sub = a[x0:x1, x0:x1]
    if diff:
        x = 0.5 + 0.5 * np.clip(sub / lim[1], -1, 1)
    else:
        x = np.clip((sub - lim[0]) / (lim[1] - lim[0] + 1e-12), 0, 1)
        x = np.sqrt(x) if stretch == "sqrt" else (np.arcsinh(10 * x) / np.arcsinh(10) if stretch == "asinh" else x)
    x = np.nan_to_num(x, nan=0.0)
    if neg:
        x = 1 - x
    im = Image.fromarray(np.flipud((x * 255).astype(np.uint8))).resize((OUT, OUT), Image.NEAREST).convert("RGB")
    if gaia is not None and len(gaia):
        d = ImageDraw.Draw(im)
        X, Y = sky_to_display(wcs, *gaia_at_epoch(gaia, frame["mjd"]), x0, x1)
        for xx, yy, g in zip(X, Y, gaia.phot_g_mean_mag):
            if 0 <= xx < OUT and 0 <= yy < OUT:
                r = max(6, 6 + (maglim - g) * 1.2)
                d.ellipse([xx - r, yy - r, xx + r, yy + r], outline=(0, 255, 255), width=2)
    return draw_extras(im, f"SPHEREx {frame['label']} ({frame['n']} img) {frame['note']}", mark, (x1 - x0) * PIX)


def render_gaia(frame, n, zoom, neg, wcs, gaia, maglim, mark):
    """Same field, same coordinates and same crop, drawn from Gaia (visible, G band)."""
    x0, x1 = crop_limits(n, zoom)
    bg, fg = ((255, 255, 255), (0, 0, 0)) if neg else ((0, 0, 0), (255, 255, 255))
    im = Image.new("RGB", (OUT, OUT), bg); d = ImageDraw.Draw(im)
    X, Y = sky_to_display(wcs, *gaia_at_epoch(gaia, frame["mjd"]), x0, x1)
    for xx, yy, g in zip(X, Y, gaia.phot_g_mean_mag):
        if 0 <= xx < OUT and 0 <= yy < OUT:
            r = max(1.5, 1.5 + (maglim - g) * 0.7)
            d.ellipse([xx - r, yy - r, xx + r, yy + r], fill=fg)
    return draw_extras(im, f"Gaia DR3 (G band) at {frame['label']}", mark, (x1 - x0) * PIX)


def to_gif(imgs, ms):
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=ms, loop=0)
    return buf.getvalue()


def show_frames(imgs, ms, name=None):
    if len(imgs) > 1:
        gif = to_gif(imgs, ms); st.image(gif)
        if name:
            st.download_button("Download GIF", gif, name, "image/gif", key=name)
    else:
        st.image(imgs[0])


def spectrum_gif(items, gids, labels, ms):
    pts = {g: [(i["wave"], i["flux"]) for i in items if i["gid"] == g and np.isfinite(i["wave"])] for g in gids}
    allf = [f for p in pts.values() for _, f in p if np.isfinite(f)]
    if not allf:
        return None
    lo, hi = np.percentile(allf, [1, 99]); pad = 0.1 * (hi - lo + 1e-9)
    imgs = []
    for k, g in enumerate(gids):
        fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=80)
        for b, (l, h) in BANDS.items():
            ax.axvspan(l, h, color="C%d" % (b - 1), alpha=0.07)
        for g2 in gids[:k]:
            if pts[g2]:
                ax.plot(*zip(*pts[g2]), "o", ms=3, color="0.75")
        if pts[g]:
            ax.plot(*zip(*pts[g]), "o", ms=5, color="crimson")
        ax.set(xlim=(0.7, 5.05), ylim=(lo - pad, hi + pad), xlabel="λ [µm]", ylabel="Flux [mJy]", title=labels[k])
        fig.tight_layout(); fig.canvas.draw()
        imgs.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB"))
        plt.close(fig)
    return to_gif(imgs, ms)


# ================================================================ persistent classification table
CLS_COLS = ["key", "ra_deg", "dec_deg", "fov_side_arcsec", "moving_object_votes", "no_moving_object_votes",
            "total_votes", "last_updated_utc", "consensus"]
LAST_COL = "I"
KEY_RE = re.compile(r"^-?\d+[.,]\d+_-?\d+[.,]\d+_\d+$")
LOCAL_CSV = "classifications.csv"


def consensus(m, nm):
    return "Moving object" if m > nm else ("No moving object" if nm > m else "Tie")


@st.cache_resource
def _worksheet():
    """Google Sheet worksheet if credentials exist in st.secrets, else None (local CSV fallback)."""
    try:
        sa, gs = dict(st.secrets["gcp_service_account"]), st.secrets["gsheet"]["sheet_id"]
    except Exception:
        return None
    import gspread
    from google.oauth2.service_account import Credentials
    creds = Credentials.from_service_account_info(
        sa, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"])
    ws = gspread.authorize(creds).open_by_key(gs).sheet1
    first = ws.row_values(1)
    if not first:                      # empty sheet -> write the titles in row 1
        ws.update(values=[CLS_COLS], range_name=f"A1:{LAST_COL}1")
    elif KEY_RE.match(first[0]):       # row 1 holds data (no titles) -> insert the titles above it
        ws.insert_row(CLS_COLS, 1)
    return ws


def _clean(df):
    for c in CLS_COLS:
        if c not in df:
            df[c] = ""
    df = df[CLS_COLS].copy()
    for c in ("ra_deg", "dec_deg"):  # Sheets with a Spanish locale return "269,45"
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    for c in ("fov_side_arcsec", "moving_object_votes", "no_moving_object_votes", "total_votes"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["consensus"] = [consensus(a, b) for a, b in zip(df.moving_object_votes, df.no_moving_object_votes)]
    return df


def load_votes():
    ws = _worksheet()
    if ws is not None:
        vals = ws.get_all_values()  # row 1 = titles; read by position, so custom titles also work
        rows = [(r + [""] * len(CLS_COLS))[:len(CLS_COLS)] for r in vals[1:] if any(r)]
        return _clean(pd.DataFrame(rows, columns=CLS_COLS))
    try:
        return _clean(pd.read_csv(LOCAL_CSV))
    except Exception:
        return _clean(pd.DataFrame(columns=CLS_COLS))


def register_vote(ra, dec, side, moving):
    key = f"{ra:.{COORD_DECIMALS}f}_{dec:.{COORD_DECIMALS}f}_{int(side)}"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    ws = _worksheet()
    if ws is not None:
        try:
            cell = ws.find(key, in_column=1)
        except Exception:
            cell = None
        if cell:
            row = (ws.row_values(cell.row) + [""] * len(CLS_COLS))
            m = int(float(row[4] or 0)) + int(moving); nm = int(float(row[5] or 0)) + int(not moving)
            ws.update(values=[[key, round(ra, 6), round(dec, 6), int(side), m, nm, m + nm, now, consensus(m, nm)]],
                      range_name=f"A{cell.row}:{LAST_COL}{cell.row}")
        else:
            ws.append_row([key, round(ra, 6), round(dec, 6), int(side), int(moving), int(not moving), 1, now,
                           consensus(int(moving), int(not moving))])
        return
    df = load_votes()
    if (df["key"] == key).any():
        i = df.index[df["key"] == key][0]
        df.loc[i, "moving_object_votes"] += int(moving)
        df.loc[i, "no_moving_object_votes"] += int(not moving)
        df.loc[i, "total_votes"] += 1
        df.loc[i, "last_updated_utc"] = now
    else:
        df.loc[len(df)] = [key, round(ra, 6), round(dec, 6), int(side), int(moving), int(not moving), 1, now, ""]
    df["consensus"] = [consensus(a, b) for a, b in zip(df.moving_object_votes, df.no_moving_object_votes)]
    df.to_csv(LOCAL_CSV, index=False)


# ================================================================ sidebar
sb = st.sidebar
sb.header("Data")
names = list_releases()
auto = sb.checkbox("Latest release automatically", True)
if auto:
    rel = latest(names)
    sb.caption("Using: " + ", ".join(rel))
    deep = sb.checkbox("Also include the Deep Fields (dr1, qr3)", False)
    colls = tuple(rel + ([r + "_deep" for r in rel] if deep else []))
else:
    colls = tuple(sb.multiselect("Releases", names, default=latest(names),
                                 help="Includes spherex_dr1_deep and spherex_qr3_deep (Deep Survey)."))

sb.header("Target")
how = sb.radio("Input", ["Name (SIMBAD)", "RA/Dec coordinates"])
obj = None
if how.startswith("Name"):
    nm = sb.text_input("Identifier", "Barnard's Star")
    obj = resolve_name(nm) if nm else None
    if obj is None:
        st.warning("Object not found in SIMBAD."); st.stop()
    ra, dec = obj["ra"], obj["dec"]
else:
    ra = sb.number_input("RA [deg]", 0.0, 360.0, 269.452077, format="%.6f")
    dec = sb.number_input("Dec [deg]", -90.0, 90.0, 4.693365, format="%.6f")

sb.header("Image")
side = float(sb.number_input("Field of view: image side [arcsec]", 60, 2000, 700, step=30,
                             help="Side length of the square image. Wider fields follow faster sources "
                                  "but take longer to download."))
area = int(side ** 2)
mode_lbl = sb.radio("Infrared bands", ["Single band", "Sum of all bands"])
mode = "band" if mode_lbl.startswith("Single") else "sum"
band = sb.selectbox("Band", list(BANDS), format_func=lambda b: f"Band {b} ({BANDS[b][0]}-{BANDS[b][1]} µm)",
                    disabled=(mode == "sum"))
all_dets = sb.checkbox("Download all bands (full spectra)", mode == "sum", disabled=(mode == "sum"))
rng = sb.date_input("Date range", (date(2025, 3, 11), date.today()))
if not isinstance(rng, (tuple, list)) or len(rng) < 2:
    sb.info("Select a start and an end date."); st.stop()
d0, d1 = rng
gap = sb.slider("Gap that separates epochs [days]", 0.5, 30.0, 3.0)
maxn = sb.slider("Max. images to download", 5, 150, 40)
bgsub = sb.checkbox("Subtract background (median) per image", True)

sb.header("Animation")
ms = sb.slider("Time between frames [ms]", 100, 3000, 700, 100)
neg = sb.checkbox("Negative (invert intensities)", False)
stretch = sb.selectbox("Stretch", ["linear", "sqrt", "asinh"])
mark = sb.checkbox("Mark center", True)
use_gaia = sb.checkbox("Compare with Gaia (visible)", False,
                       help="Shows the same field from Gaia DR3 next to the SPHEREx images.")
overlay = sb.checkbox("Overlay Gaia circles on SPHEREx", True, disabled=not use_gaia)
maglim = sb.slider("Gaia limiting G magnitude", 10.0, 21.0, 18.0, disabled=not use_gaia)
r_ap = sb.slider("Spectrum aperture [arcsec]", 6, 40, 12)

# ================================================================ target info
st.subheader(f"RA {ra:.5f}°  Dec {dec:+.5f}°")
c1, c2 = st.columns([3, 1])
with c1:
    if obj:
        st.write(f"**Object type:** {obj['info'].get('otype', '?')} – {obj['desc']}")
        st.dataframe(pd.DataFrame([obj["info"]]), hide_index=True)
with c2:
    st.link_button("Open in SIMBAD ↗", simbad_url(ra, dec, side / 120.0))
st.caption("SIMBAD URL (to reuse in other sites):")
st.code(simbad_url(ra, dec, side / 120.0), language=None)

# ================================================================ search & download
sig = (colls, round(ra, 5), round(dec, 5), area, mode, band, all_dets, str(d0), str(d1), gap, maxn, bgsub, r_ap)
if st.button("Search and generate", type="primary"):
    if not colls:
        st.error("Select at least one release."); st.stop()
    with st.spinner("Querying IRSA (SIA)…"):
        df = sia_search(ra, dec, colls)
    df = df[(df.date.dt.date >= d0) & (df.date.dt.date <= d1)]
    dets = list(BANDS) if (mode == "sum" or all_dets) else [band]
    df = df[df.det.isin(dets)].reset_index(drop=True)
    if df.empty:
        st.error("No SPHEREx images for these parameters."); st.stop()
    if len(df) > maxn:
        df = df.iloc[np.unique(np.linspace(0, len(df) - 1, maxn).astype(int))].reset_index(drop=True)
        st.info(f"Sampled {maxn} images spread over time.")
    df["gid"] = (df.t_min.diff().fillna(0) > gap).cumsum()
    n = int(max(8, round(side / PIX)))
    w = WCS(naxis=2); w.wcs.ctype = ["RA---TAN", "DEC--TAN"]; w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]; w.wcs.cdelt = [-PIX / 3600, PIX / 3600]
    results, bar = {}, st.progress(0.0, "Downloading cutouts…")
    with ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(download, r.access_url, ra, dec, side / 3600): k for k, r in df.iterrows()}
        for j, f in enumerate(as_completed(futs)):
            results[futs[f]] = f.result(); bar.progress((j + 1) / len(futs))
    items = []
    for k, r in df.iterrows():
        res = results[k]
        if "error" in res:
            continue
        wcs = WCS(fits.Header.fromstring(res["hdr"]))
        ccd = CCDData(res["data"], unit="MJy/sr", wcs=wcs)
        rp = reproject_interp((ccd.data, ccd.wcs), w, shape_out=(n, n), return_footprint=False)
        flux = aperture_flux(res["data"], res["x"], res["y"], res["scale"], r_ap, 1.5 * r_ap, 3 * r_ap)
        items.append(dict(gid=int(r.gid), mjd=float(r.t_min), det=int(r.det), wave=res["wave"],
                          rp=rp, flux=flux, meta=res["meta"], uri=r.access_url))
    st.success(f"{len(items)}/{len(df)} cutouts downloaded.")
    st.session_state.res = dict(items=items, wcs=w, sig=sig, zoom=0, ra=ra, dec=dec, area=area, side=side)

R = st.session_state.get("res")
if not R:
    st.info("Set the parameters and press **Search and generate**."); st.stop()
if R["sig"] != sig:
    st.error("⚠️ You changed parameters (field of view, band, dates, …) but the images below still come from the "
             "previous run. Press **Search and generate** to apply them.")

items = R["items"]
frames = make_frames(items, mode, band, bgsub)
labels = [f["label"] for f in frames]
n_pix = frames[0]["arr"].shape[0]

# Gaia is fetched here (not inside the button) so toggling the checkbox always takes effect.
gaia = None
if use_gaia:
    try:
        with st.spinner("Querying Gaia DR3…"):
            gaia, gaia_src = gaia_stars(round(R["ra"], 5), round(R["dec"], 5), R["side"] * 0.75 / 3600, float(maglim))
        st.caption(f"Gaia DR3 source used: {gaia_src}")
        if len(gaia) == 0:
            st.info("Gaia returned no sources in this field for the chosen magnitude limit.")
    except Exception as e:
        st.warning(f"Gaia is not available right now: {e}")
has_gaia = gaia is not None and len(gaia) > 0

# ================================================================ images (SPHEREx | Gaia)
st.header("Photometric images (FITS cutouts)")
zc = st.columns([1, 1, 6])
if zc[0].button("🔍 +"):
    R["zoom"] = min(len(ZOOMS) - 1, R["zoom"] + 1)
if zc[1].button("🔍 −"):
    R["zoom"] = max(0, R["zoom"] - 1)
zoom = ZOOMS[R["zoom"]]
zc[2].caption(f"Zoom ×{zoom} · generated field: {R['side']:.0f}″ per side ({R['area']:,} arcsec²) · "
              f"{len(frames)} epochs")
diffmode = st.selectbox("Show", ["Image", "Subtract previous epoch", "Subtract first epoch"])
stack = np.stack([f["arr"] for f in frames])
if diffmode == "Image":
    lim = tuple(np.nanpercentile(stack, [1, 99.5])); frs = frames
else:
    ref = (lambda i: frames[max(i - 1, 0)]["arr"]) if "previous" in diffmode else (lambda i: frames[0]["arr"])
    frs = [dict(f, arr=f["arr"] - ref(i)) for i, f in enumerate(frames)]
    lim = (0, 3 * float(np.nanstd(np.stack([f["arr"] for f in frs])) + 1e-9))
imgs = [render(f, lim, stretch, neg, zoom, diffmode != "Image", R["wcs"],
               gaia if (has_gaia and overlay) else None, maglim, mark) for f in frs]

if has_gaia:
    left, right = st.columns(2)
    with left:
        st.subheader("SPHEREx (near-infrared)")
        show_frames(imgs, ms, "spherex.gif")
    with right:
        st.subheader("Gaia DR3 (visible)")
        show_frames([render_gaia(f, n_pix, zoom, neg, R["wcs"], gaia, maglim, mark) for f in frames], ms, "gaia.gif")
    st.caption("Both panels show the same coordinates and field size, with Gaia positions propagated with proper "
               "motion to each frame date. Cyan circles on the SPHEREx panel mark Gaia sources. A SPHEREx source "
               "with no Gaia counterpart, or one that shifts between frames, is a moving-object candidate.")
else:
    show_frames(imgs, ms, "spherex.gif")

# ================================================================ difference A-B
st.header("Subtraction of two epochs")
ca, cb = st.columns(2)
ia = ca.selectbox("Epoch A", range(len(frames)), format_func=lambda i: labels[i], index=0)
ib = cb.selectbox("Epoch B", range(len(frames)), format_func=lambda i: labels[i], index=len(frames) - 1)
A, B = frames[ia]["arr"], frames[ib]["arr"]
D = A - B
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, arr, t in zip(axs, [A, B, D], [labels[ia], labels[ib], "A − B"]):
    if t == "A − B":
        k = 3 * np.nanstd(D) + 1e-9
        ax.imshow(arr, origin="lower", cmap="RdBu_r", vmin=-k, vmax=k)
    else:
        ax.imshow(arr, origin="lower", cmap="gray_r" if neg else "gray",
                  vmin=lim[0] if diffmode == "Image" else None, vmax=lim[1] if diffmode == "Image" else None)
    ax.set_title(t); ax.axis("off")
st.pyplot(fig); plt.close(fig)
st.caption("Fixed stars cancel out; what remains (positive in one epoch and negative in the other, like a pair) "
           "suggests a moving object. Use the same band in both epochs.")

st.subheader("Difference imaging (all epochs)")
if len(frames) < 2:
    st.info("At least two epochs are needed.")
else:
    dc1, dc2, dc3 = st.columns(3)
    refmode = dc1.selectbox("Reference built from the other epochs", ["median", "mean"],
                            help="Median is robust: a moving object barely contaminates the reference.")
    kσ = dc2.slider("Show only above N × noise (σ)", 1.0, 10.0, 3.0, 0.5)
    dshow = dc3.selectbox("Display", ["Positive and negative", "Positive only"])
    dframes, sigmas = [], []
    for i, f in enumerate(frames):
        others = np.stack([g["arr"] for j, g in enumerate(frames) if j != i])
        refim = np.nanmedian(others, axis=0) if refmode == "median" else np.nanmean(others, axis=0)
        d = f["arr"] - refim
        sig_i = 1.4826 * float(np.nanmedian(np.abs(d - np.nanmedian(d)))) + 1e-12  # robust noise (MAD)
        dframes.append(dict(f, arr=np.where(np.abs(d) > kσ * sig_i, d, 0.0), sig=sig_i))
        sigmas.append(sig_i)
    cmap_d = plt.get_cmap("RdBu_r")

    def render_diff(f):
        a = f["arr"]; x0, x1 = crop_limits(a.shape[0], zoom)
        sub = np.nan_to_num(a[x0:x1, x0:x1] / (6 * f["sig"]))
        if dshow.startswith("Positive"):
            rgb = (np.stack([np.clip(sub, 0, 1)] * 3, -1) * 255).astype(np.uint8)
        else:
            rgb = (cmap_d(np.clip(0.5 + 0.5 * sub, 0, 1))[..., :3] * 255).astype(np.uint8)
        im = Image.fromarray(np.flipud(rgb)).resize((OUT, OUT), Image.NEAREST)
        if has_gaia and overlay:
            d_ = ImageDraw.Draw(im)
            X, Y = sky_to_display(R["wcs"], *gaia_at_epoch(gaia, f["mjd"]), x0, x1)
            for xx, yy, g in zip(X, Y, gaia.phot_g_mean_mag):
                if 0 <= xx < OUT and 0 <= yy < OUT:
                    r_ = max(6, 6 + (maglim - g) * 1.2)
                    d_.ellipse([xx - r_, yy - r_, xx + r_, yy + r_], outline=(0, 255, 255), width=2)
        return draw_extras(im, f"Difference {f['label']} (>{kσ:g}σ)", mark, (x1 - x0) * PIX)

    dimgs = [render_diff(f) for f in dframes]
    gl, gr = st.columns(2)
    with gl:
        st.markdown("**One difference image per epoch** (epoch − reference of the others)")
        show_frames(dimgs, ms, "difference.gif")
    with gr:
        st.markdown("**Combined**: strongest residual over all epochs")
        comb = np.nanmax(np.stack([np.abs(f["arr"]) / (6 * f["sig"]) for f in dframes]), axis=0)
        x0, x1 = crop_limits(comb.shape[0], zoom)
        cim = Image.fromarray(np.flipud((np.clip(comb[x0:x1, x0:x1], 0, 1) * 255).astype(np.uint8)))
        st.image(draw_extras(cim.resize((OUT, OUT), Image.NEAREST).convert("RGB"), "Combined residuals", mark,
                             (x1 - x0) * PIX))
    st.caption("Fixed stars cancel when each epoch is subtracted from the combination of the others; only pixels "
               "above the noise threshold remain. A moving object appears as a source that changes position from "
               "frame to frame. Best results: single band, background subtraction on, and at least 3 epochs. "
               "Residuals on bright stars can come from differences in PSF or wavelength between epochs.")

# ================================================================ spectra
st.header("Spectra per epoch")
gids = [f["gid"] for f in frames]
sg = spectrum_gif(items, gids, labels, ms)
if sg:
    st.image(sg)
    if mode == "band" and not all_dets:
        st.caption("With a single band the spectrum covers a narrow range; enable «Download all bands» for 0.75–5 µm.")
    ea, eb = st.columns(2)
    sa = ea.selectbox("Spectrum epoch A", range(len(frames)), format_func=lambda i: labels[i], key="sa")
    sb_ = eb.selectbox("Spectrum epoch B", range(len(frames)), format_func=lambda i: labels[i], key="sb",
                       index=len(frames) - 1)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for g, c in ((gids[sa], "C0"), (gids[sb_], "C3")):
        p = sorted((i["wave"], i["flux"]) for i in items if i["gid"] == g and np.isfinite(i["wave"]))
        if p:
            ax.plot(*zip(*p), "o-", color=c, label=labels[gids.index(g)])
    ax.set(xlabel="λ [µm]", ylabel="Flux [mJy]"); ax.legend(); st.pyplot(fig); plt.close(fig)
else:
    st.info("No valid spectral data.")

with st.expander("What does a spectrum tell us about discovering a new object?"):
    st.markdown(
        "- The spectrum (0.75–5 µm) reveals the nature of the source: stars ≈ blackbody; galaxies with lines/PAH "
        "(3.3 µm); asteroids and Solar System bodies with reflected sunlight, a red slope and ice/water bands (~3 µm).\n"
        "- Comparing spectra from different epochs **in the same aperture**: a fixed source stays stable (except "
        "variables); a moving object enters and leaves the aperture, so its flux appears and disappears.\n"
        "- The most robust test is **image subtraction**: whatever does not cancel is a candidate. Then check Gaia "
        "and SIMBAD to make sure it is not a known source.")

# ================================================================ metadata + legend
st.header("Image metadata (FITS headers)")
e = st.selectbox("Epoch", range(len(frames)), format_func=lambda i: labels[i], key="meta")
rows = []
for i in items:
    if i["gid"] == gids[e]:
        rows.append({"detector": i["det"], "center λ [µm]": round(i["wave"], 4),
                     "flux [mJy]": round(i["flux"], 3), **i["meta"]})
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.markdown("""
**Column legend**

| Column | Meaning |
|---|---|
| `detector` | SPHEREx detector array (1–6); each one covers one infrared band. |
| `center λ [µm]` | Wavelength at the target pixel, from the spectral WCS (varies across the detector). |
| `flux [mJy]` | Background-subtracted aperture flux at the target (aperture set in the sidebar; background from an annulus). |
| `DETECTOR`, `OBSID` | Detector and observation identifiers from the FITS header. |
| `DATE` | Date recorded in the FITS header. |
| `PSF_FWHM` | Full width at half maximum of the point spread function. |
| `VERSION` | Processing version from the primary header. |
| `N_*` | Pixel counters from the L2 processing: number of pixels carrying each flag; keywords ending in `_NB` exclude pixels also flagged bad and those ending in `_NS` exclude pixels also flagged SOURCE. |
| `zodi_median` | Median of the zodiacal-light model in the cutout [MJy/sr]; it is **not** subtracted from the IMAGE. |
| `img_median` | Median surface brightness of the IMAGE cutout [MJy/sr]. |
| `frac_NaN` | Fraction of cutout pixels with no data (0 = complete, 1 = empty). |
""")

# ================================================================ classification
st.header("Classify this field")
st.write("Did you see an object moving relative to the background stars in this field of view?")
key_now = f"{R['ra']:.{COORD_DECIMALS}f}_{R['dec']:.{COORD_DECIMALS}f}_{int(R['side'])}"
voted = st.session_state.setdefault("voted", set())
b1, b2, _ = st.columns([1, 1, 3])
choice = None
if b1.button("Moving object present", type="primary"):
    choice = True
if b2.button("No moving object"):
    choice = False
if choice is not None:
    if key_now in voted:
        st.info("You already classified this field in this session.")
    else:
        try:
            register_vote(R["ra"], R["dec"], R["side"], choice)
            voted.add(key_now)
            st.success("Thanks! Your classification was saved.")
        except Exception as ex:
            st.error(f"Could not save the classification: {ex}")
if _worksheet() is None:
    st.caption("⚠️ No Google Sheets credentials found: votes are stored in a local CSV "
               "(not persistent on Streamlit Community Cloud).")
votes = load_votes()
st.subheader("Classification table")
st.dataframe(votes.drop(columns=["key"], errors="ignore").sort_values("total_votes", ascending=False)
             if len(votes) else votes, hide_index=True)
st.download_button("Download table (CSV)", votes.to_csv(index=False), "classifications.csv", "text/csv")
