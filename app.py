"""SPHEREx explorer: busca objetos con movimiento relativo a las estrellas de fondo.
Ejecutar:  streamlit run app.py
"""
import io, re, time, warnings, logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
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

PIX = 6.15  # arcsec/pixel de SPHEREx
BANDS = {1: (0.75, 1.11), 2: (1.11, 1.64), 3: (1.64, 2.42),
         4: (2.42, 3.82), 5: (3.82, 4.42), 6: (4.42, 5.00)}
ZOOMS = [1, 1.5, 2, 3, 4, 6, 8]
OUT = 512

st.set_page_config(page_title="SPHEREx Explorer", layout="wide")
st.title("🔭 SPHEREx Explorer – movimiento relativo a las estrellas de fondo")

# ---------------------------------------------------------------- helpers
@st.cache_data(ttl=3600)
def list_releases():
    try:
        t = Irsa.list_collections(servicetype="SIA", filter="spherex")
        names = sorted(str(c) for c in t["collection"])
    except Exception:
        names = ["spherex_dr1", "spherex_dr1_deep", "spherex_qr3", "spherex_qr3_deep"]
    return [n for n in names if re.fullmatch(r"spherex_(dr|qr)\d+", n)]


def latest(names):
    best = {}
    for n in names:
        m = re.fullmatch(r"spherex_(dr|qr)(\d+)", n)
        if m and (m[1] not in best or int(m[2]) > best[m[1]][0]):
            best[m[1]] = (int(m[2]), n)
    return [v[1] for v in best.values()]


def _simbad_astroquery(name):
    """SIMBAD vía astroquery. Se evita la consulta a /capabilities (que devuelve 404)
    fijando el límite duro a mano, y se prueban los dos espejos oficiales."""
    last = None
    for server in ("simbad.cds.unistra.fr", "simbad.cfa.harvard.edu"):
        try:
            sim = Simbad()
            try:
                sim.server = server
            except Exception:
                pass
            sim._hardlimit = 2_000_000          # <- evita tap.capabilities (el 404)
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
                ot = str(row["otype"])
                d = sim.query_tap(f"SELECT description FROM otypedef WHERE otype = '{ot}'")
                desc = str(d[0][0])
            except Exception:
                pass
            return dict(ra=float(row["ra"]), dec=float(row["dec"]), info=info, desc=desc)
        except Exception as e:
            last = e
    raise last


def _simbad_sesame(name):
    """Plan B: resolutor Sesame (CDS) -> RA/Dec; luego intenta completar info con TAP directo."""
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
    res = Irsa.query_sia(pos=(SkyCoord(ra * u.deg, dec * u.deg), 1 * u.arcsec),
                         collection=list(collections))
    t = res.to_table() if hasattr(res, "to_table") else res  # versiones nuevas ya devuelven Table
    df = pd.DataFrame({"access_url": [v.decode() if isinstance(v, bytes) else str(v) for v in t["access_url"]]})
    for c in ("t_min", "em_min", "em_max"):
        df[c] = np.ma.filled(t[c].astype(float), np.nan) if c in t.colnames else np.nan
    df["bandpass"] = [str(v) for v in t["energy_bandpassname"]] if "energy_bandpassname" in t.colnames else ""
    df["det"] = [int(m[1]) if (m := re.search(r"(\d)\s*$", b)) else 0 for b in df["bandpass"]]
    mid = (df.em_min + df.em_max) / 2
    mid = np.where(mid < 1e-3, mid * 1e6, mid)
    for i in np.where(df.det == 0)[0]:  # detector desconocido -> por longitud de onda
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
                for key in hdr:  # contadores de píxeles marcados (L2 N_*)
                    if "N_" in str(key) and isinstance(hdr[key], (int, float)):
                        meta[str(key)] = hdr[key]
                meta["zodi_mediana"] = float(np.nanmedian(zodi)) if zodi is not None else np.nan
                meta["img_mediana"] = float(np.nanmedian(data))
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


@st.cache_data(ttl=3600, show_spinner=False)
def gaia_stars(ra, dec, rad_deg, maglim):
    from astroquery.gaia import Gaia
    q = (f"SELECT TOP 2000 ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source "
         f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{rad_deg})) "
         f"AND phot_g_mean_mag < {maglim} ORDER BY phot_g_mean_mag")
    return Gaia.launch_job(q).get_results().to_pandas().fillna(0.0)


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
                note = "λ más cercana"
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


def render(frame, lim, stretch, neg, zoom, diff, wcs, gaia, maglim, mark):
    a = frame["arr"]; n = a.shape[0]
    half = max(4, int(n / zoom / 2)); c = n // 2; x0 = max(0, c - half); x1 = min(n, c + half)
    sub = a[x0:x1, x0:x1]
    if diff:
        x = 0.5 + 0.5 * np.clip(sub / lim[1], -1, 1)
    else:
        x = np.clip((sub - lim[0]) / (lim[1] - lim[0] + 1e-12), 0, 1)
        x = np.sqrt(x) if stretch == "raíz" else (np.arcsinh(10 * x) / np.arcsinh(10) if stretch == "asinh" else x)
    x = np.nan_to_num(x, nan=0.0)
    if neg:
        x = 1 - x
    im = Image.fromarray(np.flipud((x * 255).astype(np.uint8))).resize((OUT, OUT), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(im); w = x1 - x0; s = OUT / w
    if gaia is not None and len(gaia):
        dt = 2000 + (frame["mjd"] - 51544.5) / 365.25 - 2016.0
        ra = gaia.ra + gaia.pmra * dt / 3.6e6 / np.cos(np.deg2rad(gaia.dec))
        dec = gaia.dec + gaia.pmdec * dt / 3.6e6
        px, py = wcs.world_to_pixel_values(ra.values, dec.values)
        for X, Y, G in zip(px, py, gaia.phot_g_mean_mag):
            X, Y = (X - x0 + 0.5) * s, (w - (Y - x0) - 0.5) * s
            if 0 <= X < OUT and 0 <= Y < OUT:
                r = max(3, 3 + (maglim - G) * 0.9)
                d.ellipse([X - r, Y - r, X + r, Y + r], outline=(0, 200, 255))
    if mark:
        d.line([OUT / 2 - 12, OUT / 2, OUT / 2 - 5, OUT / 2], fill=(255, 60, 60))
        d.line([OUT / 2 + 5, OUT / 2, OUT / 2 + 12, OUT / 2], fill=(255, 60, 60))
    d.text((8, 8), f"{frame['label']}  ({frame['n']} img) {frame['note']}", fill=(255, 220, 0))
    return im


def to_gif(imgs, ms):
    buf = io.BytesIO()
    imgs[0].save(buf, format="GIF", save_all=True, append_images=imgs[1:], duration=ms, loop=0)
    return buf.getvalue()


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
        ax.set(xlim=(0.7, 5.05), ylim=(lo - pad, hi + pad), xlabel="λ [µm]", ylabel="Flujo [mJy]", title=labels[k])
        fig.tight_layout(); fig.canvas.draw()
        imgs.append(Image.frombuffer("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba()).convert("RGB"))
        plt.close(fig)
    return to_gif(imgs, ms)


# ---------------------------------------------------------------- sidebar
sb = st.sidebar
sb.header("Datos")
names = list_releases()
auto = sb.checkbox("Última publicación automática", True)
if auto:
    rel = latest(names)
    sb.caption("Usando: " + ", ".join(rel))
else:
    rel = sb.multiselect("Publicaciones", names, default=latest(names))
deep = sb.checkbox("Incluir Deep Fields", False)
colls = tuple(rel + ([r + "_deep" for r in rel] if deep else []))

sb.header("Objetivo")
how = sb.radio("Entrada", ["Nombre (SIMBAD)", "Coordenadas RA/Dec"])
obj = None
if how.startswith("Nombre"):
    nm = sb.text_input("Identificador", "M81")
    obj = resolve_name(nm) if nm else None
    if obj is None:
        st.warning("Objeto no encontrado en SIMBAD."); st.stop()
    ra, dec = obj["ra"], obj["dec"]
else:
    ra = sb.number_input("RA [deg]", 0.0, 360.0, 148.88822, format="%.6f")
    dec = sb.number_input("Dec [deg]", -90.0, 90.0, 69.06529, format="%.6f")

sb.header("Imagen")
area = sb.number_input("Campo de visión [arcsec²]", 3600, 4_000_000, 90000, step=10000,
                       help="Campos más amplios siguen fuentes más rápidas, pero tardan más en descargarse.")
side = float(np.sqrt(area))
mode_lbl = sb.radio("Bandas", ["Banda individual", "Suma de todas las bandas"])
mode = "band" if mode_lbl.startswith("Banda") else "sum"
band = sb.selectbox("Banda", list(BANDS), format_func=lambda b: f"Banda {b} ({BANDS[b][0]}-{BANDS[b][1]} µm)",
                    disabled=(mode == "sum"))
all_dets = sb.checkbox("Descargar todas las bandas (espectros completos)", mode == "sum", disabled=(mode == "sum"))
rng = sb.date_input("Rango de fechas", (date(2025, 3, 11), date.today()))
if not isinstance(rng, (tuple, list)) or len(rng) < 2:
    sb.info("Selecciona fecha inicial y final."); st.stop()
d0, d1 = rng
gap = sb.slider("Separar épocas si el hueco > días", 0.5, 30.0, 3.0)
maxn = sb.slider("Máx. imágenes a descargar", 5, 150, 40)
bgsub = sb.checkbox("Restar fondo (mediana) por imagen", True)

sb.header("Animación")
ms = sb.slider("Tiempo entre imágenes [ms]", 100, 3000, 700, 100)
neg = sb.checkbox("Negativo (invertir intensidades)", False)
stretch = sb.selectbox("Estiramiento", ["lineal", "raíz", "asinh"])
mark = sb.checkbox("Marcar centro", True)
use_gaia = sb.checkbox("Superponer Gaia (visible)", False)
maglim = sb.slider("Mag. G límite", 10.0, 21.0, 18.0, disabled=not use_gaia)
r_ap = sb.slider("Apertura espectro [arcsec]", 6, 40, 12)

# ---------------------------------------------------------------- objeto
st.subheader(f"RA {ra:.5f}°  Dec {dec:+.5f}°")
c1, c2 = st.columns([3, 1])
with c1:
    if obj:
        st.write(f"**Tipo de objeto:** {obj['info'].get('otype', '?')} – {obj['desc']}")
        st.dataframe(pd.DataFrame([obj["info"]]), hide_index=True)
with c2:
    st.link_button("Abrir en SIMBAD ↗", simbad_url(ra, dec, side / 120.0))
st.caption("URL de SIMBAD (para usar en otros sitios):")
st.code(simbad_url(ra, dec, side / 120.0), language=None)

# ---------------------------------------------------------------- buscar y descargar
sig = (colls, round(ra, 5), round(dec, 5), area, mode, band, all_dets, str(d0), str(d1), gap, maxn, bgsub, r_ap)
if st.button("🔎 Buscar y generar", type="primary"):
    if not colls:
        st.error("Selecciona al menos una publicación."); st.stop()
    with st.spinner("Consultando IRSA (SIA)…"):
        df = sia_search(ra, dec, colls)
    df = df[(df.date.dt.date >= d0) & (df.date.dt.date <= d1)]
    dets = list(BANDS) if (mode == "sum" or all_dets) else [band]
    df = df[df.det.isin(dets)].reset_index(drop=True)
    if df.empty:
        st.error("No hay imágenes SPHEREx para estos parámetros."); st.stop()
    if len(df) > maxn:
        df = df.iloc[np.unique(np.linspace(0, len(df) - 1, maxn).astype(int))].reset_index(drop=True)
        st.info(f"Se muestrearon {maxn} imágenes distribuidas en el tiempo.")
    df["gid"] = (df.t_min.diff().fillna(0) > gap).cumsum()
    n = int(max(8, round(side / PIX)))
    w = WCS(naxis=2); w.wcs.ctype = ["RA---TAN", "DEC--TAN"]; w.wcs.crval = [ra, dec]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]; w.wcs.cdelt = [-PIX / 3600, PIX / 3600]
    results, bar = {}, st.progress(0.0, "Descargando cutouts…")
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
    st.success(f"{len(items)}/{len(df)} cutouts descargados.")
    gaia = None
    if use_gaia:
        try:
            gaia = gaia_stars(round(ra, 5), round(dec, 5), side * 0.75 / 3600, maglim)
        except Exception as e:
            st.warning(f"Gaia no disponible: {e}")
    st.session_state.res = dict(items=items, wcs=w, gaia=gaia, sig=sig, zoom=0)

R = st.session_state.get("res")
if not R:
    st.info("Configura los parámetros y pulsa **Buscar y generar**."); st.stop()
if R["sig"] != sig:
    st.warning("Cambiaste parámetros que afectan la descarga: pulsa de nuevo **Buscar y generar**.")

items = R["items"]
frames = make_frames(items, mode, band, bgsub)
labels = [f["label"] for f in frames]

# ---------------------------------------------------------------- GIF imágenes
st.header("Imágenes fotométricas (cutouts FITS)")
zc = st.columns([1, 1, 6])
if zc[0].button("🔍 +"):
    R["zoom"] = min(len(ZOOMS) - 1, R["zoom"] + 1)
if zc[1].button("🔍 −"):
    R["zoom"] = max(0, R["zoom"] - 1)
zoom = ZOOMS[R["zoom"]]
zc[2].caption(f"Zoom ×{zoom} · {side:.0f}″ × {side:.0f}″ ({area:,} arcsec²) · {len(frames)} épocas")
diffmode = st.selectbox("Mostrar", ["Imagen", "Restar época anterior", "Restar primera época"])
stack = np.stack([f["arr"] for f in frames])
if diffmode == "Imagen":
    lim = tuple(np.nanpercentile(stack, [1, 99.5])); frs = frames
else:
    ref = (lambda i: frames[max(i - 1, 0)]["arr"]) if "anterior" in diffmode else (lambda i: frames[0]["arr"])
    frs = [dict(f, arr=f["arr"] - ref(i)) for i, f in enumerate(frames)]
    lim = (0, 3 * float(np.nanstd(np.stack([f["arr"] for f in frs])) + 1e-9))
imgs = [render(f, lim, stretch, neg, zoom, diffmode != "Imagen", R["wcs"], R["gaia"] if use_gaia else None, maglim, mark)
        for f in frs]
if len(imgs) > 1:
    gif = to_gif(imgs, ms); st.image(gif); st.download_button("Descargar GIF", gif, "spherex.gif", "image/gif")
else:
    st.image(imgs[0])
if use_gaia:
    st.caption("Círculos cian = fuentes Gaia (movidas con su movimiento propio a la fecha de cada cuadro). "
               "Un objeto SPHEREx sin círculo, o que se desplaza entre cuadros, es candidato a objeto móvil.")

# ---------------------------------------------------------------- diferencia A-B
st.header("Resta de dos épocas")
ca, cb = st.columns(2)
ia = ca.selectbox("Época A", range(len(frames)), format_func=lambda i: labels[i], index=0)
ib = cb.selectbox("Época B", range(len(frames)), format_func=lambda i: labels[i], index=len(frames) - 1)
A, B = frames[ia]["arr"], frames[ib]["arr"]
D = A - B
fig, axs = plt.subplots(1, 3, figsize=(13, 4.2))
for ax, arr, t in zip(axs, [A, B, D], [labels[ia], labels[ib], "A − B"]):
    if t == "A − B":
        k = 3 * np.nanstd(D) + 1e-9
        ax.imshow(arr, origin="lower", cmap="RdBu_r", vmin=-k, vmax=k)
    else:
        ax.imshow(arr, origin="lower", cmap="gray_r" if neg else "gray", vmin=lim[0] if diffmode == "Imagen" else None,
                  vmax=lim[1] if diffmode == "Imagen" else None)
    ax.set_title(t); ax.axis("off")
st.pyplot(fig); plt.close(fig)
st.caption("Las estrellas fijas se cancelan; lo que queda (positivo en una época y negativo en la otra, "
           "como un par) sugiere un objeto en movimiento. Usa la misma banda en ambas épocas.")

# ---------------------------------------------------------------- espectros
st.header("Espectros por época")
gids = [f["gid"] for f in frames]
sg = spectrum_gif(items, gids, labels, ms)
if sg:
    st.image(sg)
    if mode == "band" and not all_dets:
        st.caption("Con una sola banda el espectro cubre un rango estrecho; activa «todas las bandas» para 0.75–5 µm.")
    ea, eb = st.columns(2)
    sa = ea.selectbox("Espectro época A", range(len(frames)), format_func=lambda i: labels[i], key="sa")
    sb_ = eb.selectbox("Espectro época B", range(len(frames)), format_func=lambda i: labels[i], key="sb", index=len(frames) - 1)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for g, c in ((gids[sa], "C0"), (gids[sb_], "C3")):
        p = sorted((i["wave"], i["flux"]) for i in items if i["gid"] == g and np.isfinite(i["wave"]))
        if p:
            ax.plot(*zip(*p), "o-", color=c, label=labels[gids.index(g)])
    ax.set(xlabel="λ [µm]", ylabel="Flujo [mJy]"); ax.legend(); st.pyplot(fig); plt.close(fig)
else:
    st.info("Sin datos espectrales válidos.")

with st.expander("¿Qué significa el espectro para descubrir un objeto nuevo?"):
    st.markdown(
        "- El espectro (0.75–5 µm) revela la naturaleza: estrellas ≈ cuerpo negro; galaxias con líneas/PAH (3.3 µm); "
        "asteroides y objetos del Sistema Solar con luz solar reflejada, pendiente roja y bandas de hielo/agua (~3 µm).\n"
        "- Comparar espectros de épocas distintas **en la misma apertura**: una fuente fija se mantiene estable "
        "(salvo variables); un objeto móvil entra y sale de la apertura, así que su flujo aparece y desaparece.\n"
        "- Lo más robusto es la **resta de imágenes**: lo que no se cancela es candidato. Luego se verifica con Gaia "
        "y SIMBAD que no sea una fuente conocida.")

# ---------------------------------------------------------------- metadatos
st.header("Metadatos / indicadores de contaminación")
e = st.selectbox("Época", range(len(frames)), format_func=lambda i: labels[i], key="meta")
rows = []
for i in items:
    if i["gid"] == gids[e]:
        rows.append({"detector": i["det"], "λ centro [µm]": round(i["wave"], 4), "flujo [mJy]": round(i["flux"], 3), **i["meta"]})
st.dataframe(pd.DataFrame(rows), hide_index=True)
st.caption("N_* = píxeles marcados en el procesamiento L2; zodi_mediana = luz zodiacal (no restada en IMAGE); "
           "frac_NaN = píxeles sin datos.")
