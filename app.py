"""SxMOV · SPHEREx Moving Object Visualizer
Busca objetos con movimiento relativo a las estrellas de fondo en datos multi-época de SPHEREx.

Vistas (st.session_state["vista"]):
    portada  ->  hub  ->  mision | video | explorador
Ejecutar:  streamlit run app.py
"""
import io, os, re, time, base64, warnings, logging, html, mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from urllib.parse import quote_plus, quote, urlencode

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

st.set_page_config(page_title="SxMOV · SPHEREx Moving Object Visualizer", page_icon="🔭",
                   layout="wide", initial_sidebar_state="collapsed")

# El navegador (Chrome/Edge) ofrece traducir la página "en -> es" y al reescribir el DOM rompe React:
# "NotFoundError: removeChild". Se declara la página como español / no traducible.
try:
    import streamlit.components.v1 as _components
    _components.html("""<script>
    try {
      const d = window.parent.document;
      d.documentElement.lang = "es";
      d.documentElement.setAttribute("translate", "no");
      d.documentElement.classList.add("notranslate");
      if (!d.querySelector('meta[name="google"][content="notranslate"]')) {
        const m = d.createElement("meta"); m.name = "google"; m.content = "notranslate"; d.head.appendChild(m);
      }
    } catch (e) {}
    </script>""", height=0)
except Exception:
    pass

# ================================================================ CONFIGURACIÓN EDITABLE
# Todas las rutas son relativas a la carpeta de app.py (o absolutas). Si un archivo no existe,
# la interfaz usa un respaldo visual sin mostrar avisos ni rutas.
FOTO_EQUIPO = "foto_equipo.jpg"          # fondo de las vistas internas (opcional)
PANEL_ALPHA = 0.55                       # opacidad del panel oscuro sobre el fondo
FOTO_OSCURECER = 0.25                    # velo oscuro sobre la foto
VIDEO_INICIO = "Multimedia/nasa.mp4"     # video de fondo de la portada
VIDEO_PROYECTO = "Multimedia/presentacion.mp4"   # video de presentación (vista "video")
PITCH_URL = "https://www.youtube.com/"   # TODO: enlace al pitch oficial de Black Hole_Sun
LOGO_NASA = "Multimedia/logo-nasa.jpg"
LOGO_EQUIPO = "Multimedia/logo-equipo.png"
APP_URL = ""                             # URL pública de la app (para "Copiar enlace"); vacío = autodetectar
CONTACTO = {                             # TODO: completar con los datos reales del equipo
    "Equipo": "Black Hole_Sun",
    "Correo": "contacto@ejemplo.com",
    "Sede": "NASA International Space Apps Challenge — Bogotá, Colombia",
}
MISION_MEDIA = {                         # imágenes opcionales de la vista "Misión"
    "telescopio": "Multimedia/mision_telescopio.jpg",
    "hielo": "Multimedia/mision_hielo.jpg",
    "galaxias": "Multimedia/mision_galaxias.jpg",
    "inflacion": "Multimedia/mision_inflacion.jpg",
}
FUENTES = [("NASA/IPAC IRSA", "https://irsa.ipac.caltech.edu"),
           ("SIMBAD Astronomical Database", "https://simbad.cds.unistra.fr/simbad/"),
           ("ESA Gaia Archive", "https://gea.esac.esa.int/archive/")]
IRSA_SPHEREX_URL = "https://irsa.ipac.caltech.edu/Missions/spherex.html"
GAIA_ARCHIVE_URL = "https://gea.esac.esa.int/archive/"
HORIZONS_URL = "https://ssd.jpl.nasa.gov/horizons/app.html"

HERE = os.path.dirname(os.path.abspath(__file__))


def _abs(p):
    """Resuelve la ruta y es tolerante: ignora mayúsculas/minúsculas y prueba extensiones
    (.jpg/.jpeg/.png/.webp), también buscando en la carpeta de app.py si no está en su subcarpeta."""
    full = p if os.path.isabs(p) else os.path.join(HERE, p)
    if os.path.exists(full):
        return full
    stem = os.path.splitext(os.path.basename(full))[0].lower()
    exts = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".mp4"}
    for folder in dict.fromkeys([os.path.dirname(full), HERE, os.path.join(HERE, "Multimedia")]):
        try:
            for f in os.listdir(folder):
                n, e = os.path.splitext(f)
                if n.lower() == stem and e.lower() in exts:
                    return os.path.join(folder, f)
        except OSError:
            continue
    return full


def H(s):
    """Colapsa saltos de línea/indentación para que Markdown no convierta el HTML en bloque de código."""
    return re.sub(r"\s*\n\s*", " ", s.strip())


@st.cache_data(show_spinner=False)
def file_uri(path, mtime, mime):
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode()


def asset_uri(rel):
    p = _abs(rel)
    if not os.path.exists(p):
        return None
    mime = mimetypes.guess_type(p)[0] or "image/png"
    try:
        return file_uri(p, os.path.getmtime(p), mime)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def bg_data_uri(path, mtime):
    """Lee la foto, la reduce a máx. 1920 px y la devuelve como data-URI."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((1920, 1920))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def st_image(path, **kw):
    for extra in ({"width": "stretch"}, {"use_container_width": True}, {}):
        try:
            return st.image(path, **kw, **extra)
        except TypeError:
            continue


# ================================================================ ESTADO Y NAVEGACIÓN
qp = st.query_params
EXP_DEFAULTS = dict(target="M81", band=1, how="Nombre (SIMBAD)", ra_in=148.88822, dec_in=69.06529,
                    auto_rel=True, deep=False, use_gaia=False, maglim=18.0, area=90000,
                    mode_lbl="Banda individual", all_dets=False, stretch="lineal", neg=False,
                    bgsub=True, r_ap=12, rng=(date(2025, 3, 11), date.today()), gap=3.0,
                    maxn=40, ms=700, mark=True)

if "vista" not in st.session_state:
    st.session_state["vista"] = "explorador" if "target" in qp else "portada"
    if "target" in qp:
        EXP_DEFAULTS["target"] = str(qp["target"])
        try:
            b = int(qp.get("band", 1))
            EXP_DEFAULTS["band"] = b if b in BANDS else 1
        except Exception:
            pass
        st.session_state["_autorun"] = True
    try:
        st.query_params.clear()     # tras consumir un enlace compartido, F5 vuelve a la portada
    except Exception:
        pass

for _k, _v in EXP_DEFAULTS.items():       # valores iniciales (los widgets se crean sólo con key)
    st.session_state.setdefault(_k, _v)
for _k in list(EXP_DEFAULTS) + ["rel_sel"]:   # mantiene vivos los valores al navegar entre vistas
    if _k in st.session_state:
        st.session_state[_k] = st.session_state[_k]


def ir(v):
    st.session_state["vista"] = v


# ================================================================ ESTILOS GLOBALES
_foto = _abs(FOTO_EQUIPO)
BG_CSS, BG_SIZE = None, "cover"
if os.path.exists(_foto):
    try:
        BG_CSS = (f"linear-gradient(rgba(0,0,0,{FOTO_OSCURECER}), rgba(0,0,0,{FOTO_OSCURECER})), "
                  f'url("{bg_data_uri(_foto, os.path.getmtime(_foto))}")')
    except Exception:
        BG_CSS = None
if BG_CSS is None:   # respaldo: cielo estrellado en CSS
    BG_SIZE = "250px 220px, 250px 220px, 250px 220px, 250px 220px, 100% 100%, 100% 100%, auto"
    BG_CSS = ("radial-gradient(1px 1px at 20px 30px,#fff,transparent),"
              "radial-gradient(1px 1px at 90px 120px,#cfe8ff,transparent),"
              "radial-gradient(1.5px 1.5px at 160px 60px,#fff,transparent),"
              "radial-gradient(1px 1px at 230px 180px,#9cc4ff,transparent),"
              "radial-gradient(ellipse at 20% 10%,#12264d 0%,transparent 55%),"
              "radial-gradient(ellipse at 80% 90%,#2a1240 0%,transparent 55%),#02040a")

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;600&display=swap');
:root { --ac:#38bdf8; --ac2:#00d4ff; --tx:#e6edf7; --mut:#8fa3c4; --bd:rgba(56,189,248,.28); --ease:cubic-bezier(0.25,0.8,0.25,1); }

/* ---- Tipografía: Inter (UI) · Space Grotesk (títulos/marca) · JetBrains Mono (datos, coordenadas, µm, tablas) ---- */
html, body, [class*="css"], .stApp, p, li, label, button, input, textarea,
[data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], [data-baseweb="select"] { font-family:'Inter',sans-serif !important; }
h1, h2, h3, h4, [data-testid="stHeading"] *, .brand, .land-title, .card-title, .exp-title, .sec-title, .guia h3 { font-family:'Space Grotesk',sans-serif !important; }
code, pre, kbd, samp, [data-testid="stCode"] *, [data-testid="stDataFrame"], [data-testid="stTable"] *,
[data-testid="stNumberInput"] input, .coords, .top-hud, .highlight-green, .chip, .kicker, .src, .land-sub, .badge, .mono,
.card-tag, .card-st, .stat-n { font-family:'JetBrains Mono',monospace !important; }

/* ---- Limpieza de la interfaz de Streamlit ---- */
footer, [data-testid="stDecoration"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
[data-testid="stToolbar"], [data-testid="stStatusWidget"], header[data-testid="stHeader"], [data-testid="stSidebar"],
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapsedControl"] { display:none !important; }

.stApp { background:__BG__; background-size:__SIZE__; background-attachment:fixed; background-position:center; color:var(--tx); }
.block-container { max-width:1400px; padding-top:5.6rem; padding-bottom:2rem; }
[data-testid="stMain"] .block-container { background:rgba(5,10,20,__PANEL__); border:1px solid rgba(66,133,244,.15); border-radius:8px; margin-top:.5rem; }

/* ---- Animaciones ---- */
@keyframes fadeSlideUp { from { opacity:0; transform:translateY(26px); } to { opacity:1; transform:translateY(0); } }
[class*="st-key-view_"], [class*="st-key-res_"], [class*="st-key-guia_"], .landing { animation:fadeSlideUp .6s var(--ease) backwards; }
@keyframes scan { 0% { transform:translateY(-100%); } 100% { transform:translateY(100%); } }
@keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:.35; } }

/* ---- Transiciones + hover luminoso (botones y tarjetas) ---- */
div.stButton > button, div.stDownloadButton > button, div.stLinkButton > a, .card, .guia, .step,
[class*="st-key-card_"], [data-testid="stExpander"], button[data-baseweb="tab"] { transition:all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); }
div.stButton > button:hover, div.stDownloadButton > button:hover, div.stLinkButton > a:hover,
.card:hover, .step:hover, [class*="st-key-card_"]:hover {
    transform:translateY(-3px); box-shadow:0 8px 25px rgba(56,189,248,0.25); border-color:#38bdf8 !important; }

/* ---- Botones ---- */
div.stButton > button, div.stDownloadButton > button, div.stLinkButton > a {
    background-color:rgba(10,20,35,.7); color:#8ab4f8; border:1px solid rgba(66,133,244,.4); border-radius:6px;
    letter-spacing:.5px; backdrop-filter:blur(4px); }
div.stButton > button:hover, div.stDownloadButton > button:hover, div.stLinkButton > a:hover {
    background-color:rgba(25,55,95,.88); color:#fff; }
div.stButton > button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"] {
    background:linear-gradient(135deg, rgba(0,120,160,.7), rgba(56,189,248,.35)); color:#fff; border-color:var(--ac2); }

/* ---- Barra de navegación fija tipo "Misión" ---- */
.st-key-topnav { position:fixed; top:0; left:0; right:0; z-index:1000; padding:10px 28px;
    background:rgba(2,6,15,.88); backdrop-filter:blur(10px); border-bottom:1px solid var(--bd); }
.st-key-topnav [data-testid="stHorizontalBlock"] { align-items:center; gap:.6rem; }
.st-key-topnav div.stButton > button { width:100%; padding:6px 10px; font-size:.82rem; white-space:nowrap; }
.brand { font-size:1.35rem; font-weight:700; letter-spacing:.14em; color:#fff; }
.brand span { font-size:.62rem; letter-spacing:.12em; color:var(--mut); font-weight:400; margin-left:10px; text-transform:uppercase; }

/* ---- Tarjetas generales ---- */
.card { background:rgba(8,16,32,.72); border:1px solid var(--bd); border-radius:10px; padding:18px 20px; height:100%; }
.card h4 { margin:.2rem 0 .5rem; color:#fff; font-size:1.05rem; }
.card p { margin:0; color:#c5d4ee; font-size:.93rem; line-height:1.6; }
.stat-n { font-size:1.6rem; font-weight:600; color:var(--ac); }
.stat-l { color:var(--mut); font-size:.82rem; margin-top:2px; }
.sec-title { font-size:1.7rem; font-weight:700; color:#fff; margin:1.6rem 0 .4rem; letter-spacing:-.01em; }
.lede { color:#d4e0f3; font-size:1.05rem; line-height:1.7; max-width:860px; }
.kicker { letter-spacing:5px; font-size:12px; color:var(--ac2); margin-bottom:10px; }
.slot { border:1px dashed rgba(56,189,248,.45); border-radius:10px; background:rgba(8,16,32,.5); min-height:240px;
    display:flex; align-items:center; justify-content:center; text-align:center; color:var(--mut); padding:18px; font-size:.85rem; }
.step { background:rgba(8,16,32,.72); border:1px solid var(--bd); border-radius:10px; padding:18px 20px; height:100%; }
.step b { color:var(--ac); font-family:'JetBrains Mono',monospace; letter-spacing:.1em; font-size:.8rem; }
.step h4 { margin:.35rem 0; color:#fff; } .step p { color:#c5d4ee; font-size:.92rem; line-height:1.55; margin:0; }
.tl { border-left:2px solid var(--bd); margin:.6rem 0 1rem .6rem; padding-left:18px; }
.tl div { margin:0 0 14px; color:#d4e0f3; font-size:.95rem; } .tl b { color:var(--ac); font-family:'JetBrains Mono',monospace; font-size:.85rem; display:block; }

/* ---- Hub: tarjetas estilo consola espacial ---- */
[class*="st-key-card_"] { position:relative; overflow:hidden; background:linear-gradient(160deg, rgba(10,22,44,.85), rgba(4,9,20,.9));
    border:1px solid var(--bd); border-radius:12px; padding:22px 22px 16px; min-height:330px; }
[class*="st-key-card_"]::before { content:""; position:absolute; left:0; right:0; top:0; height:60px; pointer-events:none;
    background:linear-gradient(180deg, transparent, rgba(56,189,248,.09), transparent); animation:scan 4.5s linear infinite; }
[class*="st-key-card_"] div.stButton > button { width:100%; }
.card-top { display:flex; justify-content:space-between; align-items:center; margin-bottom:14px; }
.card-tag { font-size:.7rem; letter-spacing:.18em; color:var(--mut); }
.card-st { font-size:.68rem; letter-spacing:.14em; color:#00ff88; }
.card-st i { display:inline-block; width:7px; height:7px; border-radius:50%; background:#00ff88; margin-right:6px; animation:pulse 1.8s infinite; }
.card-ic { font-size:2.6rem; margin-bottom:6px; }
.card-title { font-size:1.28rem; font-weight:700; color:#fff; line-height:1.25; margin-bottom:10px; }
.card-desc { color:#c5d4ee; font-size:.92rem; line-height:1.6; min-height:92px; }

/* ---- Explorador: panel superior en flujo normal (sin position fixed/absolute) ---- */
.exp-title { font-size:1.9rem; font-weight:700; color:#fff; letter-spacing:.04em; margin-bottom:.3rem; }
.st-key-quickbar { background:rgba(8,16,32,.78); border:1px solid var(--bd); border-radius:10px; padding:12px 14px 4px; margin-bottom:.4rem; }
button[data-baseweb="tab"] { font-size:.9rem; }
.top-hud { text-align:right; color:#fff; font-size:12px; letter-spacing:1px; padding:6px 4px; }
.highlight-green { color:#00ff88; font-weight:bold; }
.coords { font-size:1.45rem; font-weight:600; color:var(--tx); margin:.4rem 0 .6rem; }
[data-testid="stExpander"] { background:rgba(5,10,20,.55); border:1px solid rgba(66,133,244,.35) !important; border-radius:8px; }

/* ---- Guía astrofísica en vivo ---- */
.guia { background:linear-gradient(150deg, rgba(10,24,50,.88), rgba(5,10,22,.9)); border:1px solid var(--bd); border-radius:12px; padding:20px 22px; margin:.8rem 0; }
.guia h3 { margin:.5rem 0 .6rem; color:#fff; font-size:1.45rem; }
.guia .lead { color:#e3ebf7; font-size:1rem; line-height:1.7; margin:0 0 14px; }
.guia-grid { display:grid; grid-template-columns:repeat(3, 1fr); gap:14px; }
.guia-grid div { background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.08); border-radius:8px; padding:12px 14px; color:#c5d4ee; font-size:.9rem; line-height:1.6; }
.guia-grid b { display:block; color:var(--ac); font-family:'JetBrains Mono',monospace; font-size:.74rem; letter-spacing:.12em; margin-bottom:6px; }
.chip { display:inline-block; font-size:.7rem; letter-spacing:.12em; color:var(--ac2); border:1px solid var(--bd); border-radius:20px; padding:3px 10px; margin-right:8px; }
@media (max-width:900px) { .guia-grid { grid-template-columns:1fr; } }

/* ---- Footer institucional ---- */
.inst-footer { margin-top:2.5rem; padding:16px 26px; border:1px solid var(--bd); border-radius:10px; background:rgba(3,8,18,.82);
    backdrop-filter:blur(8px); display:grid; grid-template-columns:auto 1fr auto; gap:26px; align-items:center; }
.inst-footer.fixed { position:fixed; left:0; right:0; bottom:0; z-index:50; margin:0; border-radius:0; border-width:1px 0 0 0; }
.inst-logos { display:flex; align-items:center; gap:18px; }
.inst-logos img { height:var(--lh); width:auto; object-fit:contain; border-radius:6px; }
.inst-logos .sep { height:var(--lh); border-left:1px solid rgba(255,255,255,.2); }
.logo-fb { height:var(--lh); display:flex; align-items:center; padding:0 14px; border:1px solid var(--bd); border-radius:6px; color:#fff; font-size:.72rem; letter-spacing:.1em; text-align:center; font-family:'JetBrains Mono',monospace; }
.inst-mid { text-align:center; color:#c5d4ee; font-size:.84rem; line-height:1.7; }
.inst-mid b { color:#fff; }
.inst-mid a { color:var(--ac); text-decoration:none; margin:0 8px; border-bottom:1px dotted rgba(56,189,248,.5); }
.inst-mid a:hover { color:#fff; }
.inst-co { display:flex; align-items:center; gap:10px; color:#fff; font-size:.8rem; letter-spacing:.12em; font-family:'JetBrains Mono',monospace; white-space:nowrap; }
.flag { display:inline-block; width:34px; height:22px; border-radius:3px; background:linear-gradient(180deg,#fcd116 0 50%,#003893 50% 75%,#ce1126 75% 100%); box-shadow:0 0 0 1px rgba(255,255,255,.25); }
@media (max-width:900px) { .inst-footer { grid-template-columns:1fr; justify-items:center; text-align:center; } }
</style>
"""
st.markdown(H(BASE_CSS.replace("__BG__", BG_CSS).replace("__SIZE__", BG_SIZE).replace("__PANEL__", str(PANEL_ALPHA))),
            unsafe_allow_html=True)

# ================================================================ COMPONENTES COMUNES


def footer_html(logo_h, titulo, fixed=False):
    def _logo(rel, alt, fallback):
        uri = asset_uri(rel)
        return f'<img src="{uri}" alt="{alt}">' if uri else f'<span class="logo-fb">{fallback}</span>'
    logos = _logo(LOGO_NASA, "NASA International Space Apps Challenge", "NASA<br>SPACE APPS") \
        + '<span class="sep"></span>' + _logo(LOGO_EQUIPO, "Black Hole_Sun", "Black<br>Hole_Sun")
    links = " ".join(f'<a href="{u_}" target="_blank" rel="noopener">{n}</a>' for n, u_ in FUENTES)
    mail = html.escape(CONTACTO.get("Correo", ""))
    contact = f' · ✉ <a href="mailto:{mail}">{mail}</a>' if mail else ""
    return H(f'''
    <div class="inst-footer {'fixed' if fixed else ''}" style="--lh:{logo_h}px">
        <div class="inst-logos">{logos}</div>
        <div class="inst-mid">
            <b>{titulo}</b><br>
            Fuentes de datos: {links}<br>
            Equipo <b>Black Hole_Sun</b>{contact}
        </div>
        <div class="inst-co"><span class="flag"></span> COLOMBIA · SEDE BOGOTÁ</div>
    </div>''')


_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)


def _contacto_body():
    for k, v in CONTACTO.items():
        st.markdown(f"**{k}:** {v}")
    st.caption("Datos del equipo Black Hole_Sun · NASA International Space Apps Challenge.")


if _dialog:
    @_dialog("Contacto del equipo Black Hole_Sun")
    def contacto_dialog():
        _contacto_body()


def contacto_button():
    if _dialog:
        if st.button("✉ Contacto del equipo", key="btn_contacto"):
            contacto_dialog()
    else:
        with st.expander("✉ Contacto del equipo"):
            _contacto_body()


def topnav():
    v = st.session_state["vista"]
    with st.container(key="topnav"):
        c = st.columns([2.4, 1.6, 1.3, 1.3, 1.6])
        c[0].markdown(H('<div class="brand">🔭 SxMOV<span>SPHEREx Moving Object Visualizer</span></div>'), unsafe_allow_html=True)
        c[1].button("← Volver al Hub" if v != "hub" else "← Portada", key="nav_back", on_click=ir,
                    args=("hub" if v != "hub" else "portada",))
        for col, (k, lbl) in zip(c[2:], [("mision", "🛰️ Misión"), ("video", "🎬 Video"), ("explorador", "🔭 Explorador")]):
            col.button(lbl, key=f"nav_{k}", on_click=ir, args=(k,), type="primary" if v == k else "secondary")


# ================================================================ VISTA 1 · PORTADA
def vista_portada():
    _stars = [("radial-gradient(1px 1px at 40px 60px,#fff,transparent)", 300),
              ("radial-gradient(1px 1px at 200px 150px,#cfe8ff,transparent)", 300),
              ("radial-gradient(1px 1px at 120px 250px,#fff,transparent)", 300),
              ("radial-gradient(1.5px 1.5px at 100px 80px,#fff,transparent)", 450),
              ("radial-gradient(1.5px 1.5px at 300px 300px,#9cc4ff,transparent)", 450),
              ("radial-gradient(1.5px 1.5px at 400px 150px,#fff,transparent)", 450),
              ("radial-gradient(2px 2px at 150px 200px,#fff,transparent)", 600),
              ("radial-gradient(2px 2px at 450px 450px,#ffd9a0,transparent)", 600)]
    _neb = ["radial-gradient(ellipse at 20% 15%,rgba(40,80,160,.45) 0%,transparent 55%)",
            "radial-gradient(ellipse at 80% 85%,rgba(120,40,160,.35) 0%,transparent 55%)"]
    _img = ",".join([g for g, _ in _stars] + _neb)
    _size = ",".join([f"{t}px {t}px" for _, t in _stars] + ["100% 100%"] * len(_neb))
    _p0 = ",".join(["0 0"] * (len(_stars) + len(_neb)))
    _p1 = ",".join([f"-{t}px {t}px" for _, t in _stars] + ["0 0"] * len(_neb))
    _vpath = _abs(VIDEO_INICIO)
    _vid_ok = os.path.exists(_vpath)

    _anim_css = """.stApp { background-color:#02040a !important; background-image:__IMG__ !important;
        background-size:__SIZE__ !important; background-repeat:repeat !important;
        background-attachment:fixed !important; animation:drift 90s linear infinite; }
        @keyframes drift { from { background-position:__P0__; } to { background-position:__P1__; } }"""
    _video_css = """.stApp { background:#02040a !important; }
        .st-key-landing_video { height:0; overflow:visible; }
        .st-key-landing_video video { position:fixed !important; top:0; left:0; width:100vw !important; height:100vh !important;
            object-fit:cover; z-index:0; pointer-events:none; }
        .st-key-landing_video video::-webkit-media-controls { display:none !important; }
        .veil { position:fixed; inset:0; z-index:1; pointer-events:none;
            background:radial-gradient(ellipse at center, rgba(0,0,0,.30) 0%, rgba(0,0,0,.68) 100%); }"""
    _planet = '<div class="planet"><div class="ring"></div><div class="moon"></div></div>'
    css = H("""<style>
        __ANIM__
        [data-testid="stMain"] .block-container { background:transparent !important; border:none !important; padding-bottom:2rem; padding-top:3rem; }
        .landing { min-height:56vh; display:flex; flex-direction:column; align-items:center; justify-content:center;
            text-align:center; padding-top:5vh; position:relative; z-index:2; }
        .land-title { font-size:clamp(52px, 9vw, 110px); font-weight:700; letter-spacing:.15em; padding-left:.15em; line-height:1.05;
            background:linear-gradient(90deg,#ffffff,#8ab4f8 60%,#00d4ff); -webkit-background-clip:text; background-clip:text;
            color:transparent; filter:drop-shadow(0 4px 16px rgba(0,0,0,.6)); }
        .land-sub { font-size:clamp(13px,1.7vw,19px); font-weight:600; color:#c5d4ee; margin-top:12px; letter-spacing:3px; text-transform:uppercase; }
        .land-desc { max-width:780px; margin-top:26px; font-size:16.5px; line-height:1.7; color:#e3ebf7; padding:18px 24px;
            background:rgba(5,10,20,.5); border:1px solid rgba(66,133,244,.3); border-radius:10px; backdrop-filter:blur(4px); }
        .land-desc p { margin:0 0 .9em 0; } .land-desc p:last-child { margin-bottom:0; }
        .land-desc p.lead { font-size:1.12em; font-weight:500; color:#fff; } .land-desc strong { color:#fff; font-weight:600; }
        .src { margin-top:18px; font-size:12px; letter-spacing:2px; color:#7f93b5; }
        .st-key-start_wrap { align-items:center; margin-top:1.2rem; position:relative; z-index:2; }
        .st-key-start_wrap button { padding:14px 56px; font-size:17px; letter-spacing:4px; }
        .planet { position:fixed; right:7vw; bottom:24vh; width:170px; height:170px; border-radius:50%; z-index:1; pointer-events:none;
            opacity:.92; animation:bob 9s ease-in-out infinite;
            background:radial-gradient(circle at 30% 30%,#7ab8ff,#1a3a7a 55%,#050b1c 100%);
            box-shadow:0 0 70px rgba(80,150,255,.35), inset -22px -22px 44px rgba(0,0,0,.65); }
        .planet .ring { position:absolute; left:-48%; top:34%; width:196%; height:32%; border-radius:50%;
            border:2px solid rgba(160,200,255,.55); transform:rotate(-18deg); }
        .planet .moon { position:absolute; width:14px; height:14px; border-radius:50%; background:#dfe9ff; left:50%; top:50%; margin:-7px;
            animation:orbit 14s linear infinite; box-shadow:0 0 10px #fff; }
        @keyframes bob { 0%,100% { transform:translateY(0); } 50% { transform:translateY(-14px); } }
        @keyframes orbit { from { transform:rotate(0deg) translateX(140px) rotate(0deg); } to { transform:rotate(360deg) translateX(140px) rotate(-360deg); } }
        @media (max-width:900px) { .planet { display:none; } }
    </style>""").replace("__ANIM__", _video_css if _vid_ok else _anim_css) \
        .replace("__IMG__", _img).replace("__SIZE__", _size).replace("__P0__", _p0).replace("__P1__", _p1)
    st.markdown(css, unsafe_allow_html=True)

    st.markdown(H(f"""
    {'<div class="veil"></div>' if _vid_ok else _planet}
    <div class="landing">
        <div class="kicker">NASA · SPHEREx · ANÁLISIS MULTI-ÉPOCA</div>
        <div class="land-title">SxMOV</div>
        <div class="land-sub">SPHEREX MOVING OBJECT VISUALIZER</div>
        <div class="land-desc">
            <p class="lead">The universe is not static: every SPHEREx observation is a <strong>footprint in time</strong>.</p>
            <p>SxMOV transforms <strong>spectroscopic data</strong> from the <strong>SPHEREx space telescope</strong> into an <strong>interactive laboratory for celestial motion</strong>. Query any coordinate or catalog (<strong>SIMBAD</strong>), align and subtract <strong>multi-epoch observations</strong> from <strong>IRSA</strong> to reveal <strong>asteroids, comets, or high-proper-motion stars</strong> hidden against the cosmic background, cross-match them with <strong>Gaia</strong> astrometry, and analyze their <strong>infrared spectral signature (0.75–5 μm)</strong> in seconds.</p>
        </div>
        <div class="src">DATOS: NASA/IPAC IRSA · SIMBAD · GAIA</div>
    </div>"""), unsafe_allow_html=True)

    if _vid_ok:
        with st.container(key="landing_video"):
            try:
                st.video(_vpath, loop=True, autoplay=True, muted=True)
            except TypeError:
                st.video(_vpath)
    with st.container(key="start_wrap"):
        st.button("▶  INICIAR EXPERIENCIA / EXPLORAR HUB →", key="start_btn", type="primary", on_click=ir, args=("hub",))
    st.markdown(footer_html(60, "NASA Space Apps Challenge — Sede Bogotá"), unsafe_allow_html=True)


# ================================================================ VISTA 2 · HUB
def vista_hub():
    st.markdown(H("""<div style="text-align:center; margin:.4rem 0 1.6rem;">
        <div class="kicker">CENTRO DE MANDO</div>
        <div class="exp-title">Elige tu punto de partida</div>
        <div class="lede" style="margin:0 auto;">Tres puertas al mismo proyecto: la historia del telescopio, nuestro video de presentación
        y el laboratorio donde se analizan los datos.</div></div>"""), unsafe_allow_html=True)
    cards = [("mision", "🛰️", "Conoce la Misión SPHEREx", "MISIÓN · 01",
              "Un telescopio espacial que fotografía todo el cielo en 102 colores infrarrojos para buscar hielo de agua, moléculas de la vida y la historia del cosmos."),
             ("video", "🎬", "Video de Presentación del Proyecto", "VIDEO · 02",
              "Mira el pitch del equipo Black Hole_Sun y cómo SxMOV convierte datos espectroscópicos en un laboratorio de movimiento celeste."),
             ("explorador", "🔭", "Interfaz de SxMOV Explorer", "LABORATORIO · 03",
              "Entra al laboratorio interactivo: consulta cualquier objeto, alinea y resta épocas y compara su espectro de 0.75 a 5 µm.")]
    for col, (k, ic, title, tag, desc) in zip(st.columns(3), cards):
        with col:
            with st.container(key=f"card_{k}"):
                st.markdown(H(f"""<div class="card-top"><span class="card-tag">{tag}</span><span class="card-st"><i></i>ONLINE</span></div>
                    <div class="card-ic">{ic}</div><div class="card-title">{title}</div><div class="card-desc">{desc}</div>"""),
                            unsafe_allow_html=True)
                st.button("ENTRAR →", key=f"enter_{k}", on_click=ir, args=(k,))


# ================================================================ VISTA 3 · MISIÓN
def media_slot(key, caption):
    rel = MISION_MEDIA.get(key, "")
    if rel and os.path.exists(_abs(rel)):
        st_image(_abs(rel), caption=caption)
    else:
        st.markdown(H(f'<div class="slot">🖼️ Espacio para imagen<br>{caption}<br><span class="mono">{html.escape(rel)}</span></div>'),
                    unsafe_allow_html=True)


def tema(key, titulo, texto, img_caption, invertido=False):
    a, b = st.columns([1.2, 1])
    txt = H(f'<div class="sec-title" style="font-size:1.35rem;margin-top:.6rem">{titulo}</div><div class="lede">{texto}</div>')
    if invertido:
        with a: media_slot(key, img_caption)
        with b: st.markdown(txt, unsafe_allow_html=True)
    else:
        with a: st.markdown(txt, unsafe_allow_html=True)
        with b: media_slot(key, img_caption)


def vista_mision():
    st.markdown(H("""<div class="kicker">MISIÓN SPHEREx · NASA</div>
        <div class="exp-title" style="font-size:2.3rem">El Universo en 102 bandas espectrales </div>
        <div class="lede">SPHEREx no apunta a un objeto: barre todo el cielo y, en cada punto, registra un espectro. Con ello nacen preguntas sorprendentes:
        ¿cómo empezó el universo?, ¿cómo se formaron las galaxias?, ¿dónde está el agua que hará posible la vida?</div>"""), unsafe_allow_html=True)
    st.write("")
    stats = [("0.75–5 µm", "rango de longitudes de onda"), ("102", "bandas espectrales (6 detectores × 17 canales)"),
             ("4", "mapas de todo el cielo en 2 años"), ("20 cm", "apertura del telescopio")]
    for col, (n, l) in zip(st.columns(4), stats):
        col.markdown(H(f'<div class="card"><div class="stat-n">{n}</div><div class="stat-l">{l}</div></div>'), unsafe_allow_html=True)

        st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

    st.markdown('<div class="sec-title" translate="no">¿Qué es SPHEREx?</div>', unsafe_allow_html=True)
    tema("telescopio", "El mapa infrarrojo más colorido del cosmos",
         "SPHEREx es un observatorio espacial de la NASA que escanea todo el cielo cada seis meses para construir el primer catálogo tridimensional "
         "del universo en 102 colores infrarrojos. Al analizar cientos de millones de galaxias, estrellas y asteroides, la misión descubre las huellas "
         "químicas de los ingredientes para la vida, mide el brillo acumulado de las primeras galaxias y revela los secretos detrás del origen del cosmos.",
         "Imagen del telescopio SPHEREx")



    st.markdown('<div class="sec-title">Tres grandes preguntas científicas</div>', unsafe_allow_html=True)
    tema("hielo", "❄️ Hielo de agua y moléculas biogénicas",
         "En las nubes frías donde nacen estrellas y planetas, los átomos se pegan a granos de polvo y forman hielos: agua (H₂O), dióxido de carbono (CO₂), "
         "monóxido de carbono (CO) y moléculas orgánicas. Cada hielo absorbe luz en longitudes de onda precisas —el agua cerca de 3 µm—. SPHEREx leerá esas "
         "huellas en millones de estrellas de fondo para mapear dónde están los ingredientes de la vida antes de que existan los planetas.",
         "Nube molecular con hielos")
    tema("galaxias", "🌌 La historia de las galaxias",
         "Sumando la luz de todas las galaxias —incluidas las demasiado débiles para verlas una a una— SPHEREx mide la luz de fondo extragaláctica: el "
         "inventario total de luz producida desde las primeras estrellas. Además ubica cientos de millones de galaxias en tres dimensiones gracias a su corrimiento al rojo.",
         "Mapa de galaxias", invertido=True)
    tema("inflacion", "💥 La inflación cósmica",
         "Una fracción de segundo después del Big Bang, el universo se expandió de forma explosiva. Las huellas de ese instante quedaron escritas en cómo se "
         "distribuyen las galaxias a gran escala. SPHEREx medirá esa distribución con una precisión sin precedentes para poner a prueba las teorías de la inflación.",
         "Inflación y estructura a gran escala")

    st.markdown('<div class="sec-title">Historia de la misión</div>', unsafe_allow_html=True)
    st.markdown(H("""<div class="tl">
        <div><b>11 MAR 2025</b>Lanzamiento de SPHEREx.</div>
        <div><b>1 MAY 2025</b>Comienzan las operaciones nominales.</div>
        <div><b>CADA SEMANA APROX.</b>Los nuevos datos llegan al archivo NASA/IPAC IRSA y quedan disponibles para la comunidad.</div>
        <div><b>2 AÑOS</b>Duración de la misión primaria: cuatro barridos completos del cielo.</div>
        <div><b>HOY</b>SxMOV usa esos datos públicos para buscar objetos que se mueven entre una observación y otra.</div></div>"""),
        unsafe_allow_html=True)
    st.button("🔭 Ver cómo SxMOV usa estos datos →", key="mision_to_exp", type="primary", on_click=ir, args=("explorador",))


# ================================================================ VISTA 4 · VIDEO
def vista_video():
    st.markdown(H("""<div class="kicker">PRESENTACIÓN · BLACK HOLE_SUN</div>
        <div class="exp-title">SxMOV en acción</div>"""), unsafe_allow_html=True)
    a, b = st.columns([1.7, 1])
    with a:
        vp = _abs(VIDEO_PROYECTO)
        if os.path.exists(vp):
            st.video(vp)
        else:
            st.markdown(H(f'<div class="slot" style="min-height:360px">🎬 Video de presentación<br>Coloca el archivo en <span class="mono">{html.escape(VIDEO_PROYECTO)}</span></div>'),
                        unsafe_allow_html=True)
    with b:
        st.markdown(H("""<div class="card"><h4>Pitch oficial del equipo</h4>
            <p>La presentación completa de Black Hole_Sun para el NASA International Space Apps Challenge — Sede Bogotá.</p></div>"""),
                    unsafe_allow_html=True)
        st.link_button("▶ Ver el pitch oficial", PITCH_URL)
    st.markdown('<div class="sec-title">Cómo SxMOV resuelve el reto</div>', unsafe_allow_html=True)
    steps = [("PASO 01", "Consultar", "Escribe un nombre (SIMBAD) o unas coordenadas. SxMOV busca en IRSA todas las observaciones SPHEREx que cubren ese punto."),
             ("PASO 02", "Alinear y restar", "Las imágenes de cada época se reproyectan a una misma rejilla y se restan. Lo fijo se cancela; lo que se mueve deja un par positivo–negativo."),
             ("PASO 03", "Cruzar y analizar", "Se superponen las estrellas de Gaia con su movimiento propio y se extrae el espectro de 0.75 a 5 µm de cada época.")]
    for col, (t, h, p) in zip(st.columns(3), steps):
        col.markdown(H(f'<div class="step"><b>{t}</b><h4>{h}</h4><p>{p}</p></div>'), unsafe_allow_html=True)
    st.write("")
    st.button("🔭 Probar el explorador →", key="video_to_exp", type="primary", on_click=ir, args=("explorador",))


# ================================================================ HELPERS CIENTÍFICOS (sin cambios)
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
    """SIMBAD vía astroquery. Se evita la consulta a /capabilities (404) y se prueban los dos espejos oficiales."""
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


@st.cache_data(ttl=3600, show_spinner=False)
def identify_coords(ra, dec):
    """Identifica (si existe) la fuente SIMBAD más cercana a unas coordenadas, para alimentar la guía."""
    try:
        import pyvo
        adql = ("SELECT TOP 1 main_id, otype, sp_type, plx_value, pmra, pmdec FROM basic "
                f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{5/3600}))")
        for server in ("simbad.cds.unistra.fr", "simbad.cfa.harvard.edu"):
            try:
                r = pyvo.dal.TAPService(f"https://{server}/simbad/sim-tap").run_sync(adql).to_table()
                if len(r):
                    info = {k: str(r[0][k]) for k in r.colnames if str(r[0][k]) not in ("--", "")}
                    return dict(ra=ra, dec=dec, info=info, desc="")
            except Exception:
                continue
    except Exception:
        pass
    return None


def simbad_url(ra, dec, radius_arcmin):
    c = SkyCoord(ra * u.deg, dec * u.deg)
    s = c.to_string("hmsdms", sep=" ", precision=6, alwayssign=True)
    return ("https://simbad.cfa.harvard.edu/simbad/sim-coo?Coord=%09" + quote_plus(s) +
            "&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none"
            f"&Radius={radius_arcmin:.2f}&Radius.unit=arcmin&submit=submit+query&CoordList=")


def _sia_una_coleccion(ra, dec, coleccion, intentos=4):
    """Consulta SIA de UNA colección con reintentos (IRSA a veces responde 502/503 con HTML)."""
    err = None
    for k in range(intentos):
        try:
            res = Irsa.query_sia(pos=(SkyCoord(ra * u.deg, dec * u.deg), 1 * u.arcsec), collection=coleccion)
            return res.to_table() if hasattr(res, "to_table") else res
        except Exception as e:
            err = e
            time.sleep(2 * (k + 1))
    raise err


@st.cache_data(ttl=600, show_spinner=False)
def sia_search(ra, dec, collections):
    """Devuelve (df, fallidas). Consulta cada colección por separado para que un 502 no tumbe todo."""
    tablas, fallidas, ultimo_error = [], [], None
    for c in collections:
        try:
            t = _sia_una_coleccion(ra, dec, c)
            if len(t):
                tablas.append(t)
        except Exception as e:
            fallidas.append(c); ultimo_error = e
    if not tablas:
        if fallidas and len(fallidas) == len(collections):
            raise RuntimeError(f"IRSA no respondió ({type(ultimo_error).__name__}). Intenta de nuevo en unos minutos.")
        return pd.DataFrame(columns=["access_url", "t_min", "em_min", "em_max", "bandpass", "det", "date"]), fallidas
    parts = []
    for t in tablas:
        d = pd.DataFrame({"access_url": [v.decode() if isinstance(v, bytes) else str(v) for v in t["access_url"]]})
        for c in ("t_min", "em_min", "em_max"):
            d[c] = np.ma.filled(t[c].astype(float), np.nan) if c in t.colnames else np.nan
        d["bandpass"] = [str(v) for v in t["energy_bandpassname"]] if "energy_bandpassname" in t.colnames else ""
        parts.append(d)
    df = pd.concat(parts, ignore_index=True).drop_duplicates("access_url")
    df["det"] = [int(m[1]) if (m := re.search(r"(\d)\s*$", b)) else 0 for b in df["bandpass"]]
    mid = (df.em_min + df.em_max) / 2
    mid = np.where(mid < 1e-3, mid * 1e6, mid)
    for i in np.where(df.det == 0)[0]:
        for k, (lo, hi) in BANDS.items():
            if lo <= mid[i] <= hi:
                df.loc[i, "det"] = k
    df = df.dropna(subset=["t_min"]).sort_values("t_min").reset_index(drop=True)
    df["date"] = Time(df.t_min.values, format="mjd").to_datetime()
    return df, fallidas


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
                for key in hdr:
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
        if stretch == "raíz":
            x = np.sqrt(x)
        elif stretch == "asinh":
            x = np.arcsinh(10 * x) / np.arcsinh(10)
        elif stretch == "log":
            x = np.log1p(1000 * x) / np.log1p(1000)
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


# ================================================================ GUÍA ASTROFÍSICA + ENLACES
GUIA = {
    "xb": dict(icon="⚡", titulo="Binaria de rayos X",
               que="Dos astros que bailan muy juntos: una estrella normal y un objeto compacto —una estrella de neutrones o un agujero negro— que le arranca gas como un sumidero cósmico. "
                   "Ese gas cae en espiral formando un disco de acreción que se calienta a millones de grados y emite rayos X.",
               mov="No esperes verla «caminar» entre las estrellas de fondo: está a miles de años luz y su movimiento propio es minúsculo en meses. Lo que sí cambia es su brillo: "
                   "estas fuentes tienen estallidos y variabilidad. Compara épocas buscando que se encienda o se apague, no que se desplace.",
               firma="Continuo de la estrella compañera más un exceso infrarrojo del disco o de un chorro (jet). Busca líneas de hidrógeno (Paschen, Brackett, Brγ a 2.17 µm), "
                     "helio a 1.083 µm y, si la compañera es fría, las bandas de CO cerca de 2.3 µm."),
    "agn": dict(icon="🌀", titulo="Núcleo activo de galaxia / cuásar",
                que="Un agujero negro supermasivo devorando materia en el centro de una galaxia, tan brillante que puede eclipsar a todas las estrellas de su anfitriona.",
                mov="Fijo contra el fondo (está a distancias cosmológicas), pero suele variar de brillo en semanas o años. Un cambio de flujo sin desplazamiento delata a un núcleo activo.",
                firma="Continuo en ley de potencias con exceso de polvo caliente hacia 3–5 µm y líneas anchas de hidrógeno (Hα, Paschen) desplazadas al rojo según la distancia."),
    "galaxia": dict(icon="🌌", titulo="Galaxia",
                    que="Una isla de cientos de miles de millones de estrellas, gas y polvo. En el infrarrojo cercano el polvo se vuelve casi transparente y vemos la estructura estelar real: "
                        "el bulbo central, los brazos espirales y las regiones donde nacen estrellas.",
                    mov="Quieta para nuestros ojos: a esas distancias ninguna galaxia se desplaza de forma medible entre épocas. Si algo cambia en la imagen, lo interesante sería una supernova o una fuente superpuesta.",
                    firma="Emisión de PAH (moléculas orgánicas) en 3.3 µm, la «joroba» de estrellas viejas cerca de 1.6 µm y, si forma estrellas, líneas de hidrógeno como Paschen-α (1.87 µm) y Brackett-α (4.05 µm), "
                          "corridas hacia el rojo según la distancia."),
    "pm": dict(icon="💨", titulo="Estrella de alto movimiento propio",
               que="Una estrella que se mueve de forma notable por el cielo, porque está muy cerca de nosotros o porque viaja muy rápido por la Galaxia.",
               mov="Sí se desplaza respecto al fondo, pero con los píxeles de 6.15″ de SPHEREx sólo las de movimiento propio más alto (del orden de segundos de arco por año) cambian de posición de forma apreciable en meses. "
                   "Activa Gaia: sus círculos se corren con el movimiento propio de cada época.",
               firma="Las cálidas muestran un continuo tipo cuerpo negro. Las enanas frías: vapor de agua (1.4 y 1.9 µm), CO (2.3 y 4.6 µm) y, en las más frías, metano en 1.6, 2.2 y 3.3 µm."),
    "star": dict(icon="⭐", titulo="Estrella",
                 que="Una esfera de plasma que brilla por fusión nuclear. Su temperatura define su color y su espectro.",
                 mov="Casi estática frente al fondo durante unos meses, salvo estrellas muy cercanas. Los cambios de brillo pueden indicar variabilidad (pulsantes, eclipsantes).",
                 firma="Un continuo suave tipo cuerpo negro con absorción de H, de CO (2.3 µm) y de agua en las estrellas frías; si hay polvo circunestelar, exceso en 3–5 µm."),
    "sso": dict(icon="☄️", titulo="Cuerpo del Sistema Solar",
                que="Un asteroide, cometa o planeta menor: restos rocosos y helados de la formación del Sol hace unos 4 600 millones de años.",
                mov="Es el protagonista de SxMOV: se mueve rápido frente a las estrellas de fondo. Entre épocas aparece en una posición y desaparece de la anterior; al restar imágenes deja un par positivo–negativo.",
                firma="Luz solar reflejada con pendiente roja, bandas de hidratación y agua cerca de 3 µm y, desde ~3.5 µm, emisión térmica propia. En cometas, CO₂ (4.27 µm) y CO (4.67 µm) del gas que liberan."),
    "otro": dict(icon="🔭", titulo="Objeto astronómico",
                 que="SIMBAD lo clasifica con un código propio que no tiene una guía específica todavía. Aun así, el método es el mismo: observar cómo cambia entre épocas.",
                 mov="Compara épocas con la resta de imágenes: si no se cancela, algo cambió de posición o de brillo.",
                 firma="Examina el espectro de 0.75 a 5 µm: un continuo suave sugiere una estrella; picos y hombros marcados, gas, polvo u hielos."),
}


def classify(otype, name):
    o = (otype or "").strip()
    nm = (name or "").strip()
    if o.startswith(("XB", "HXB", "LXB")) or o in ("X", "XRB"):
        return "xb"
    if o in ("QSO", "AGN", "Sy1", "Sy2", "Sy", "Bla", "BLL", "OVV", "LINER", "LIN", "Q?", "AG?"):
        return "agn"
    if "PM*" in o:
        return "pm"
    if o == "G" or o.endswith("G") or o in ("GiG", "GiP", "GiC", "BiC", "SBG", "H2G", "LSB", "EmG", "rG", "PaG", "IG"):
        return "galaxia"
    if o in ("Pl", "Asteroid", "Comet", "MPl", "SSO") or re.match(r"^\(?\d+\)?\s+[A-Za-z]", nm) \
            or re.match(r"^\d+[PCD]/", nm) or re.match(r"^[PCDXAI]/\d{4}", nm):
        return "sso"
    if o.endswith("*") or o == "Star":
        return "star"
    return "otro"


def link_list(ra, dec, side_arcsec, main_id, key):
    fov = max(0.05, 2 * side_arcsec / 3600.0)
    simbad = (f"https://simbad.cfa.harvard.edu/simbad/sim-id?Ident={quote(main_id)}" if main_id
              else simbad_url(ra, dec, side_arcsec / 120.0))
    aladin = f"https://aladin.cds.unistra.fr/AladinLite/?target={quote(f'{ra:.6f} {dec:+.6f}')}&fov={fov:.3f}&survey=P%2FDSS2%2Fcolor"
    out = [("🔗 Ver en SIMBAD", simbad), ("🔗 Consultar en NASA/IPAC IRSA", IRSA_SPHEREX_URL),
           ("🔗 Explorar en Aladin Lite", aladin), ("🔗 Archivo ESA Gaia", GAIA_ARCHIVE_URL)]
    if key == "sso":
        out.append(("🔗 JPL Horizons", HORIZONS_URL))
    return out


def share_url(target, band):
    base = APP_URL
    if not base:
        try:
            base = str(getattr(st.context, "url", "") or "")
        except Exception:
            base = ""
        if not base:
            try:
                h = st.context.headers
                host = h.get("Host", "")
                base = f"{h.get('X-Forwarded-Proto', 'http')}://{host}" if host else ""
            except Exception:
                base = ""
    return f"{base.split('?')[0]}?{urlencode({'target': target, 'band': band})}"


def guia_card(obj, ra, dec):
    info = {k.lower(): v for k, v in (obj["info"] if obj else {}).items()}
    otype, main_id = info.get("otype", ""), info.get("main_id", "")
    key = classify(otype, main_id or st.session_state.get("target", ""))
    g = GUIA[key]
    nombre = html.escape(main_id) if main_id else f"RA {ra:.4f}°, Dec {dec:+.4f}°"
    st.markdown(H(f"""<div class="guia">
        <span class="chip">GUÍA ASTROFÍSICA EN VIVO</span><span class="chip">otype: {html.escape(otype) if otype else 'sin clasificar'}</span>
        <h3>{g['icon']} {nombre} · {g['titulo']}</h3>
        <p class="lead">{g['que']}</p>
        <div class="guia-grid">
            <div><b>MOVIMIENTO ESPERADO</b>{g['mov']}</div>
            <div><b>FIRMAS EN 0.75–5 µm</b>{g['firma']}</div>
            <div><b>CÓMO VERLO EN SxMOV</b>Elige la banda, pulsa <i>Analizar</i> y recorre las épocas en el GIF y en la resta A − B.
            {'Activa Gaia para descartar estrellas conocidas.' if key in ('pm', 'star', 'sso') else 'Compara el flujo de la misma apertura entre épocas.'}</div>
        </div></div>"""), unsafe_allow_html=True)
    return key, main_id


# ================================================================ VISTA 5 · EXPLORADOR
def vista_explorador():
    ss = st.session_state
    st.markdown(H("""<div class="kicker">LABORATORIO INTERACTIVO · ANÁLISIS MULTI-ÉPOCA</div>
        <div class="exp-title">SxMOV Explorer</div>"""), unsafe_allow_html=True)

    # --- Panel de control superior, en flujo normal (barra de búsqueda rápida + 3 pestañas) ---
    bar = st.container(key="quickbar")
    tabs = st.tabs(["🎯 Objetivo y Catálogos", "🖼️ Imagen y Bandas", "⏱️ Tiempo y Animación"])
    hud_top = st.empty()
    names = list_releases()
    ss.setdefault("rel_sel", latest(names))

    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            how = st.radio("Entrada del objetivo", ["Nombre (SIMBAD)", "Coordenadas RA/Dec"], key="how")
            ra_in = st.number_input("RA [deg]", 0.0, 360.0, format="%.6f", key="ra_in", disabled=how.startswith("Nombre"))
            dec_in = st.number_input("Dec [deg]", -90.0, 90.0, format="%.6f", key="dec_in", disabled=how.startswith("Nombre"))
        with c2:
            auto = st.checkbox("Última publicación automática (spherex_dr1 / qr3)", key="auto_rel")
            if auto:
                rel = latest(names)
                st.caption("Usando: " + ", ".join(rel))
            else:
                rel = st.multiselect("Publicaciones", names, key="rel_sel")
            deep = st.checkbox("Incluir Campos Profundos (Deep Fields)", key="deep")
        with c3:
            use_gaia = st.checkbox("Superponer Gaia (visible)", key="use_gaia")
            maglim = st.slider("Mag. G límite", 10.0, 21.0, key="maglim", disabled=not use_gaia)
    colls = tuple(rel + ([r + "_deep" for r in rel] if deep else []))

    with tabs[1]:
        c1, c2, c3 = st.columns(3)
        with c1:
            area = st.number_input("Campo de visión [arcsec²]", 3600, 4_000_000, step=10000, key="area",
                                   help="Campos más amplios siguen fuentes más rápidas, pero tardan más en descargarse.")
            r_ap = st.slider("Apertura espectro [arcsec]", 6, 40, key="r_ap")
        with c2:
            mode_lbl = st.radio("Bandas", ["Banda individual", "Suma de todas las bandas"], key="mode_lbl")
            mode = "band" if mode_lbl.startswith("Banda") else "sum"
            all_dets = st.checkbox("Descarga completa 0.75–5 µm (todas las bandas)", key="all_dets", disabled=(mode == "sum"))
        with c3:
            stretch = st.selectbox("Estiramiento", ["lineal", "log", "raíz", "asinh"], key="stretch")
            neg = st.checkbox("Negativo (invertir intensidades)", key="neg")
            bgsub = st.checkbox("Restar mediana (fondo) por imagen", key="bgsub")
    side = float(np.sqrt(area))

    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        with c1:
            rng = st.date_input("Rango de fechas", key="rng")
            gap = st.slider("Separar épocas si el hueco > días", 0.5, 30.0, key="gap")
        with c2:
            maxn = st.slider("Máx. imágenes a descargar", 5, 150, key="maxn")
        with c3:
            ms = st.slider("Velocidad: tiempo entre imágenes [ms]", 100, 3000, step=100, key="ms")
            mark = st.checkbox("Marcar centro", key="mark")

    with bar:
        qa, qb, qc = st.columns([4, 2.4, 1.3])
        qa.text_input("Nombre SIMBAD o RA/Dec", key="target", label_visibility="collapsed",
                      placeholder="Nombre SIMBAD o RA/Dec  (ej. M81 · SS 433 · 148.888 69.065)")
        band = qb.selectbox("Banda espectral", list(BANDS), key="band", label_visibility="collapsed", disabled=(mode == "sum"),
                            format_func=lambda b: f"Banda {b} ({BANDS[b][0]}–{BANDS[b][1]} µm)")
        go = qc.button("🔎 Analizar", type="primary", key="analizar")
    go = go or bool(ss.pop("_autorun", False))

    hud_top.markdown(H('<div class="top-hud">SPHEREX LATEST DATA &nbsp;·&nbsp; Datos en uso: '
                       f'<span class="highlight-green">{", ".join(colls) if colls else "—"}</span></div>'), unsafe_allow_html=True)

    # --- Resolución del objetivo ---
    obj, target_str = None, ""
    if how.startswith("Coord"):
        ra, dec = float(ra_in), float(dec_in)
        target_str = f"{ra:.5f} {dec:.5f}"
        obj = identify_coords(round(ra, 5), round(dec, 5))
    else:
        q = ss["target"].strip()
        m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)[\s,;]+([+-]?\d+(?:\.\d+)?)", q)
        if m and 0 <= float(m[1]) <= 360 and -90 <= float(m[2]) <= 90:
            ra, dec = float(m[1]), float(m[2])
            obj = identify_coords(round(ra, 5), round(dec, 5))
        elif not q:
            st.info("Escribe un nombre (SIMBAD) o unas coordenadas RA/Dec en la barra de búsqueda."); return
        else:
            obj = resolve_name(q)
            if obj is None:
                st.warning("Objeto no encontrado en SIMBAD."); return
            ra, dec = obj["ra"], obj["dec"]
        target_str = q

    st.markdown(f'<div class="coords" translate="no">RA {ra:.5f}°&nbsp;&nbsp;Dec {dec:+.5f}°</div>', unsafe_allow_html=True)

    # --- Guía astrofísica en vivo (cambia con el objeto) ---
    with st.container(key="guia_" + re.sub(r"[^A-Za-z0-9]", "", f"{ra:.4f}{dec:.4f}")):
        gkey, main_id = guia_card(obj, ra, dec)
    links = link_list(ra, dec, side, main_id, gkey)
    cols = st.columns(len(links) + 1)
    for col, (lbl, url) in zip(cols, links):
        col.link_button(lbl, url)
    if cols[-1].button("🔗 Copiar enlace de esta observación", key="btn_share"):
        ss["_share"] = True
    if ss.get("_share"):
        st.caption("Copia este enlace (icono de la derecha) para abrir esta búsqueda exacta:")
        st.code(share_url(target_str, band), language=None)
    if obj and obj.get("info"):
        with st.expander("Ficha técnica SIMBAD"):
            if obj.get("desc"):
                st.write(f"**Tipo de objeto:** {obj['info'].get('otype', '?')} – {obj['desc']}")
            st.dataframe(pd.DataFrame([obj["info"]]), hide_index=True)

    # --- Buscar y descargar ---
    sig = (colls, round(ra, 5), round(dec, 5), area, mode, band, all_dets, str(rng), gap, maxn, bgsub, r_ap)
    if go:
        if not colls:
            st.error("Selecciona al menos una publicación."); return
        if not isinstance(rng, (tuple, list)) or len(rng) < 2:
            st.info("Selecciona fecha inicial y final."); return
        d0, d1 = rng
        try:
            with st.spinner("Consultando IRSA (SIA)…"):
                df, fallidas = sia_search(ra, dec, colls)
        except Exception as e:
            st.error(f"⚠️ {e}  (Servidor de IRSA temporalmente no disponible — error 502/503. No es un fallo de la app.)")
            return
        if fallidas:
            st.warning("IRSA no respondió para: " + ", ".join(fallidas) + ". Se muestran solo las demás colecciones.")
        df = df[(df.date.dt.date >= d0) & (df.date.dt.date <= d1)]
        dets = list(BANDS) if (mode == "sum" or all_dets) else [band]
        df = df[df.det.isin(dets)].reset_index(drop=True)
        if df.empty:
            st.error("No hay imágenes SPHEREx para estos parámetros."); return
        if len(df) > maxn:
            df = df.iloc[np.unique(np.linspace(0, len(df) - 1, maxn).astype(int))].reset_index(drop=True)
            st.info(f"Se muestrearon {maxn} imágenes distribuidas en el tiempo.")
        df["gid"] = (df.t_min.diff().fillna(0) > gap).cumsum()
        n = int(max(8, round(side / PIX)))
        w = WCS(naxis=2); w.wcs.ctype = ["RA---TAN", "DEC--TAN"]; w.wcs.crval = [ra, dec]
        w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]; w.wcs.cdelt = [-PIX / 3600, PIX / 3600]
        results, pbar = {}, st.progress(0.0, "Descargando cutouts…")
        with ThreadPoolExecutor(8) as ex:
            futs = {ex.submit(download, r.access_url, ra, dec, side / 3600): k for k, r in df.iterrows()}
            for j, f in enumerate(as_completed(futs)):
                results[futs[f]] = f.result(); pbar.progress((j + 1) / len(futs))
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
        ss["res"] = dict(items=items, wcs=w, gaia=gaia, sig=sig, zoom=0, rid=int(time.time() * 1000))

    R = ss.get("res")
    if not R:
        st.info("Configura los parámetros y pulsa **Analizar**."); return
    if R["sig"] != sig:
        st.warning("Cambiaste parámetros que afectan la descarga: pulsa de nuevo **Analizar**.")

    items = R["items"]
    if not items:
        st.error("No se pudo descargar ningún cutout. Intenta de nuevo o cambia los parámetros."); return

    with st.container(key=f"res_{R['rid']}"):
        frames = make_frames(items, mode, band, bgsub)
        labels = [f["label"] for f in frames]

        # ---- GIF de imágenes ----
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

        # ---- Resta A−B ----
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

        # ---- Espectros ----
        st.header("Espectros por época")
        gids = [f["gid"] for f in frames]
        sg = spectrum_gif(items, gids, labels, ms)
        if sg:
            st.image(sg)
            if mode == "band" and not all_dets:
                st.caption("Con una sola banda el espectro cubre un rango estrecho; activa «Descarga completa» para 0.75–5 µm.")
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

        # ---- Metadatos ----
        st.header("Metadatos / indicadores de contaminación")
        e = st.selectbox("Época", range(len(frames)), format_func=lambda i: labels[i], key="meta")
        rows = []
        for i in items:
            if i["gid"] == gids[e]:
                rows.append({"detector": i["det"], "λ centro [µm]": round(i["wave"], 4), "flujo [mJy]": round(i["flux"], 3), **i["meta"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        st.caption("N_* = píxeles marcados en el procesamiento L2; zodi_mediana = luz zodiacal (no restada en IMAGE); "
                   "frac_NaN = píxeles sin datos.")


# ================================================================ DESPACHADOR
VISTAS = {"hub": vista_hub, "mision": vista_mision, "video": vista_video, "explorador": vista_explorador}
vista = st.session_state["vista"]
if vista not in VISTAS and vista != "portada":
    vista = st.session_state["vista"] = "portada"

if vista == "portada":
    vista_portada()
else:
    topnav()
    with st.container(key=f"view_{vista}"):
        VISTAS[vista]()
    # Footer institucional al final del contenido de cada vista (aparece al hacer scroll hasta abajo)
    st.markdown(footer_html(55, "NASA International Space Apps Challenge — Sede Bogotá"), unsafe_allow_html=True)
