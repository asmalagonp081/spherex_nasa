"""SxMOV · SPHEREx Moving Object Visualizer
Busca objetos con movimiento relativo a las estrellas de fondo en datos multi-época de SPHEREx.

Vistas (st.session_state["vista"]):
    portada  ->  hub  ->  mision | video | explorador
Ejecutar:  streamlit run app.py
"""
# --------------------------------------------------------------------
# Librerías
# --------------------------------------------------------------------
import io, os, re, time, base64, warnings, logging, html, mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
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
         4: (2.42, 3.82), 5: (3.82, 4.42), 6: (4.42, 5.00)} #micrometros
ZOOMS = [1, 1.5, 2, 3, 4, 6, 8]
OUT = 512
COORD_DECIMALS = 4 # Coordenadas iguales hasta 4 decimales (~0.36") cuentan como el mismo campo

st.set_page_config(page_title="SxMOV · SPHEREx Moving Object Visualizer", page_icon="🔭",
                   layout="wide", initial_sidebar_state="collapsed")

# Evita que el navegador traduzca automáticamente la página y rompa el DOM de Streamlit,
# sin usar st.components.v1.html (que está deprecado en versiones nuevas).
st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

# ================================================================ 
# Elementos configurables del diseño de la página
# ================================================================ 
# ================================================================ 
# Todas las rutas son relativas a la carpeta de app.py (o absolutas). Si un archivo no existe,
# la interfaz usa un respaldo visual sin mostrar avisos ni rutas.
FOTO_EQUIPO = "foto_equipo.jpg"          # fondo de las vistas internas (opcional)
PANEL_ALPHA = 0.55                       # opacidad del panel oscuro sobre el fondo
FOTO_OSCURECER = 0.25                    # velo oscuro sobre la foto
VIDEO_INICIO = "Multimedia/nasa.mp4"     # video de fondo de la portada
VIDEO_PROYECTO = "Multimedia/presentacion.mp4"   # video de presentación (vista "video")
PITCH_URL = "https://www.youtube.com/watch?v=PH9TjoZ9zuE"   # TODO: enlace al pitch oficial de Black Hole_Sun
LOGO_NASA = "Multimedia/logo-nasa.jpg"
LOGO_EQUIPO = "Multimedia/logo-equipo.png"
APP_URL = "https://spherexnasa-spaceappschallenge.streamlit.app"                             # URL pública de la app (para "Copiar enlace"); vacío = autodetectar
CONTACTO = {
    "Team": "Black Hole_Sun",
    "Email": "saramirez@udistrital.edu.co",
    "Location": "NASA International Space Apps Challenge — Bogotá, Colombia",
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

# Se utiliza para buscar los archivos anteriormente definidos en la carpeta de app.py y sus subcarpetas, ignorando mayúsculas/minúsculas y probando extensiones de imagen comunes.
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


# ================================================================ STATE AND NAVIGATION
qp = st.query_params
# Configuración inicial con Barnard's Star por defecto y parámetros en inglés
EXP_DEFAULTS = dict(
    target="Barnard's Star", band=1, how="Name (SIMBAD)",
    ra_in=269.452077, dec_in=4.693365,
    ra_hms="17h 57m 48.498s", dec_dms="+04° 41' 36.11\"",
    auto_rel=True, deep=False, use_gaia=False, overlay=True, maglim=18.0,
    side=700, area=490000, mode_lbl="Single band", all_dets=False,
    stretch="linear", neg=False, bgsub=True, r_ap=12,
    rng=(date(2025, 3, 11), date.today()), gap=3.0, maxn=20, ms=700, mark=True
)

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
        st.query_params.clear()
    except Exception:
        pass

for _k, _v in EXP_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
for _k in list(EXP_DEFAULTS) + ["rel_sel"]:
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

def footer_html(logo_h, titulo, fixed=False):
    def _logo(rel, alt, fallback):
        uri = asset_uri(rel)
        return f'<img src="{uri}" alt="{alt}">' if uri else f'<span class="logo-fb">{fallback}</span>'
    logos = _logo(LOGO_NASA, "NASA International Space Apps Challenge", "NASA<br>SPACE APPS") \
        + '<span class="sep"></span>' + _logo(LOGO_EQUIPO, "Black Hole_Sun", "Black<br>Hole_Sun")
    links = " ".join(f'<a href="{u_}" target="_blank" rel="noopener">{n}</a>' for n, u_ in FUENTES)
    mail = html.escape(CONTACTO.get("Email", CONTACTO.get("Correo", "")))
    contact = f' · ✉ <a href="mailto:{mail}">{mail}</a>' if mail else ""
    return H(f'''
    <div class="inst-footer {'fixed' if fixed else ''}" style="--lh:{logo_h}px">
        <div class="inst-logos">{logos}</div>
        <div class="inst-mid">
            <b>{titulo}</b><br>
            Data sources: {links}<br>
            Team <b>Black Hole_Sun</b>{contact}
        </div>
        <div class="inst-co"><span class="flag"></span> COLOMBIA · BOGOTÁ HUB</div>
    </div>''')


_dialog = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)


def _contacto_body():
    for k, v in CONTACTO.items():
        st.markdown(f"**{k}:** {v}")
    st.caption("Team Black Hole_Sun · NASA International Space Apps Challenge.")


if _dialog:
    @_dialog("Team Black Hole_Sun Contact")
    def contacto_dialog():
        _contacto_body()


def contacto_button():
    if _dialog:
        if st.button("✉ Team Contact", key="btn_contacto"):
            contacto_dialog()
    else:
        with st.expander("✉ Team Contact"):
            _contacto_body()


def topnav():
    v = st.session_state["vista"]
    with st.container(key="topnav"):
        c = st.columns([2.4, 1.6, 1.3, 1.3, 1.6])
        c[0].markdown(H('<div class="brand">🔭 SxMOV<span>SPHEREx Moving Object Visualizer</span></div>'), unsafe_allow_html=True)
        c[1].button("← Back to Hub" if v != "hub" else "← Home", key="nav_back", on_click=ir,
                    args=("hub" if v != "hub" else "portada",))
        for col, (k, lbl) in zip(c[2:], [("mision", "🛰️ Mission"), ("video", "🎬 Video"), ("explorador", "🔭 Explorer")]):
            col.button(lbl, key=f"nav_{k}", on_click=ir, args=(k,), type="primary" if v == k else "secondary")


# ===================================================================================
# PORTADA - VISTA 1 
# ===================================================================================
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
        <div class="kicker">NASA · SPHEREx · MULTI-EPOCH ANALYSIS</div>
        <div class="land-title">SxMOV</div>
        <div class="land-sub">SPHEREX MOVING OBJECT VISUALIZER</div>
        <div class="land-desc">
            <p class="lead">The universe is not static: every SPHEREx observation is a <strong>footprint in time</strong>.</p>
            <p>SxMOV transforms <strong>spectroscopic data</strong> from the <strong>SPHEREx space telescope</strong> into an <strong>interactive laboratory for celestial motion</strong>. Query any coordinate or catalog (<strong>SIMBAD</strong>), align and subtract <strong>multi-epoch observations</strong> from <strong>IRSA</strong> to reveal <strong>asteroids, comets, or high-proper-motion stars</strong> hidden against the cosmic background, cross-match them with <strong>Gaia</strong> astrometry, and analyze their <strong>infrared spectral signature (0.75–5 μm)</strong> in seconds.</p>
        </div>
        <div class="src">DATA: NASA/IPAC IRSA · SIMBAD · GAIA</div>
    </div>"""), unsafe_allow_html=True)

    if _vid_ok:
        with st.container(key="landing_video"):
            try:
                st.video(_vpath, loop=True, autoplay=True, muted=True)
            except TypeError:
                st.video(_vpath)
    with st.container(key="start_wrap"):
        st.button("▶  START EXPERIENCE →", key="start_btn", type="primary", on_click=ir, args=("hub",))
    st.markdown(footer_html(60, "NASA Space Apps Challenge — Bogotá Hub"), unsafe_allow_html=True)


# ===================================================================================
# VISTA 2 - HUB
# ===================================================================================
def vista_hub():
    st.markdown(H("""<div style="text-align:center; margin:.4rem 0 1.6rem;">
        <div class="kicker">COMMAND CENTER</div>
        <div class="exp-title">Choose your starting point</div>
        <div class="lede" style="margin:0 auto;">Three gateways to the project: the story of the space telescope, our project video presentation,
        and the interactive laboratory where the data is analyzed.</div></div>"""), unsafe_allow_html=True)
    cards = [("mision", "🛰️", "Discover the SPHEREx Mission", "MISSION · 01",
              "A space telescope mapping the entire sky in 102 infrared colors to search for water ice, biogenic molecules, and the history of the cosmos."),
             ("video", "🎬", "Project Presentation Video", "VIDEO · 02",
              "Watch the official pitch by Team Black Hole_Sun and see how SxMOV turns spectroscopic data into a celestial motion laboratory."),
             ("explorador", "🔭", "SxMOV Explorer Interface", "LABORATORY · 03",
              "Enter the interactive laboratory: query any object, align and subtract epochs, and compare its 0.75 to 5 µm spectrum.")]
    for col, (k, ic, title, tag, desc) in zip(st.columns(3), cards):
        with col:
            with st.container(key=f"card_{k}"):
                st.markdown(H(f"""<div class="card-top"><span class="card-tag">{tag}</span><span class="card-st"><i></i>ONLINE</span></div>
                    <div class="card-ic">{ic}</div><div class="card-title">{title}</div><div class="card-desc">{desc}</div>"""),
                            unsafe_allow_html=True)
                st.button("ENTER →", key=f"enter_{k}", on_click=ir, args=(k,))


# ===================================================================================
# VISTA 3 - DETALLES DE LA MISIÓN SPHEREX
# ===================================================================================
def media_slot(key, caption):
    rel = MISION_MEDIA.get(key, "")
    if rel and os.path.exists(_abs(rel)):
        st_image(_abs(rel), caption=caption)
    else:
        st.markdown(H(f'<div class="slot">🖼️ Image placeholder<br>{caption}<br><span class="mono">{html.escape(rel)}</span></div>'),
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
    st.markdown(H("""<div class="kicker">SPHEREx MISSION · NASA</div>
        <div class="exp-title" style="font-size:2.3rem">The Universe in 102 spectral bands</div>
        <div class="lede">SPHEREx does not just point at single targets: it sweeps the entire sky and records a spectrum at every position. This unlocks
        profound questions: How did the universe begin? How did galaxies form? Where are the water and biogenic molecules that make life possible?</div>"""), unsafe_allow_html=True)
    st.write("")
    stats = [("0.75–5 µm", "wavelength range"), ("102", "spectral bands (6 detectors × 17 channels)"),
             ("4", "all-sky maps in 2 years"), ("20 cm", "telescope aperture")]
    for col, (n, l) in zip(st.columns(4), stats):
        col.markdown(H(f'<div class="card"><div class="stat-n">{n}</div><div class="stat-l">{l}</div></div>'), unsafe_allow_html=True)
        st.markdown('<meta name="google" content="notranslate">', unsafe_allow_html=True)

    st.markdown('<div class="sec-title" translate="no">What is SPHEREx?</div>', unsafe_allow_html=True)
    tema("telescopio", "The most colorful infrared map of the cosmos",
         "SPHEREx is a NASA space observatory scanning the entire sky every six months to build the first three-dimensional map "
         "of the universe in 102 infrared colors. By analyzing hundreds of millions of galaxies, stars, and asteroids, the mission uncovers "
         "the chemical fingerprints of life's ingredients, measures the cumulative glow of the first galaxies, and probes the origins of the cosmos.",
         "SPHEREx space telescope")

    st.markdown('<div class="sec-title">Three major scientific questions</div>', unsafe_allow_html=True)
    tema("hielo", "❄️ Water ice and biogenic molecules",
         "In cold molecular clouds where stars and planets are born, atoms freeze onto dust grains to form ices: water (H₂O), carbon dioxide (CO₂), "
         "carbon monoxide (CO), and organic molecules. Each ice absorbs light at specific wavelengths—water near 3 µm. SPHEREx reads these "
         "fingerprints across millions of background stars to map the ingredients of life before planets even form.",
         "Molecular cloud with interstellar ices")
    tema("galaxias", "🌌 The history of galaxies",
         "By summing the light of all galaxies—including those too faint to be resolved individually—SPHEREx measures the extragalactic background light: "
         "the total inventory of light emitted since the first stars. It also maps hundreds of millions of galaxies in 3D via their spectroscopic redshift.",
         "3D galaxy distribution map", invertido=True)
    tema("inflacion", "💥 Cosmic inflation",
         "A fraction of a second after the Big Bang, the universe expanded exponentially. The imprints of that instant are encoded in the large-scale "
         "three-dimensional distribution of galaxies. SPHEREx measures this distribution with unprecedented precision to test models of cosmic inflation.",
         "Cosmic inflation and large-scale structure")

    st.markdown('<div class="sec-title">Mission timeline</div>', unsafe_allow_html=True)
    st.markdown(H("""<div class="tl">
        <div><b>11 MAR 2025</b>SPHEREx launch.</div>
        <div><b>1 MAY 2025</b>Nominal science operations begin.</div>
        <div><b>WEEKLY</b>New data arrives at the NASA/IPAC IRSA archive and becomes publicly available to the community.</div>
        <div><b>2 YEARS</b>Primary mission duration: four complete all-sky surveys.</div>
        <div><b>TODAY</b>SxMOV uses this public data to discover and track objects moving between observations.</div></div>"""),
        unsafe_allow_html=True)
    st.button("🔭 See how SxMOV uses this data →", key="mision_to_exp", type="primary", on_click=ir, args=("explorador",))


# ===================================================================================
# VISTA 4 - VIDEO DE PRESENTACIÓN DEL PROYECTO
# ===================================================================================
def vista_video():
    st.markdown(H("""<div class="kicker">PRESENTATION · BLACK HOLE_SUN</div>
        <div class="exp-title">SxMOV in action</div>"""), unsafe_allow_html=True)
    a, b = st.columns([1.7, 1])
    with a:
        vp = _abs(VIDEO_PROYECTO)
        if os.path.exists(vp):
            st.video(vp)
        else:
            st.markdown(H(f'<div class="slot" style="min-height:360px">🎬 Presentation video<br>Place the file at <span class="mono">{html.escape(VIDEO_PROYECTO)}</span></div>'),
                        unsafe_allow_html=True)
    with b:
        st.markdown(H("""<div class="card"><h4>Official Team Pitch</h4>
            <p>Full presentation by Team Black Hole_Sun for the NASA International Space Apps Challenge — Bogotá Hub.</p></div>"""),
                    unsafe_allow_html=True)
        st.link_button("▶ Watch official pitch", PITCH_URL)
    st.markdown('<div class="sec-title">How SxMOV solves the challenge</div>', unsafe_allow_html=True)
    steps = [("STEP 01", "Query", "Enter a target name (SIMBAD) or coordinates (h m s / ° ' ''). SxMOV searches IRSA for all SPHEREx observations covering that patch of sky."),
             ("STEP 02", "Align & Subtract", "Images from each epoch are reprojected onto a common WCS grid and subtracted. Fixed stars cancel out; moving objects leave a positive–negative pair."),
             ("STEP 03", "Cross-Match & Analyze", "Gaia DR3 stars are overlaid with their proper motions propagated to each epoch, while extracting the 0.75 to 5 µm spectrum on the fly.")]
    for col, (t, h, p) in zip(st.columns(3), steps):
        col.markdown(H(f'<div class="step"><b>{t}</b><h4>{h}</h4><p>{p}</p></div>'), unsafe_allow_html=True)
    st.write("")
    st.button("🔭 Launch SxMOV Explorer →", key="video_to_exp", type="primary", on_click=ir, args=("explorador",))

# ===================================================================================
# VISTA 5 - EXPLORADOR DE DATOS SPHEREx
# ===================================================================================

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

@st.cache_data(ttl=600, show_spinner=False)
def sia_search(ra, dec, collections):
    """
    Método exacto de app_sin_interfaz.py:
    Consulta IRSA (SIA) con radio puntual de 1 arcsec para todas las colecciones seleccionadas.
    """
    # Consultamos todas las colecciones en una sola llamada rápida (igual que en app_sin_interfaz.py)
    res = Irsa.query_sia(pos=(SkyCoord(ra * u.deg, dec * u.deg), 1 * u.arcsec), collection=list(collections))
    t = res.to_table() if hasattr(res, "to_table") else res
    if len(t) == 0:
        return pd.DataFrame(columns=["access_url", "t_min", "em_min", "em_max", "bandpass", "det", "date"])
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


GAIA_COLS = ["ra", "dec", "pmra", "pmdec", "phot_g_mean_mag"]
VIZIER_TAPS = ("https://tapvizier.cds.unistra.fr/TAPVizieR/tap", "https://vizier.cfa.harvard.edu/TAPVizieR/tap")


def _gaia_irsa(ra, dec, rad, maglim):
    """Consulta Gaia DR3 desde el servicio TAP de IRSA."""
    cats = Irsa.list_catalogs(filter="gaia")
    name = next((k for k, v in cats.items() if "dr3" in (k + v).lower() and "source" in (k + v).lower()), None)
    if name is None:
        raise RuntimeError("no Gaia DR3 table at IRSA")
    q = (f"SELECT TOP 2000 ra, dec, pmra, pmdec, phot_g_mean_mag FROM {name} "
         f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{rad})) "
         f"AND phot_g_mean_mag < {maglim} ORDER BY phot_g_mean_mag")
    return Irsa.query_tap(q).to_table().to_pandas()


def _gaia_vizier_tap(url):
    """Genera una función de consulta para los espejos TAP de VizieR (CDS y Harvard CfA)."""
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
    """Consulta Gaia DR3 a través de MAST (archive.stsci.edu)."""
    from astroquery.mast import Catalogs
    try:
        Catalogs.TIMEOUT = 30
    except Exception:
        pass
    t = Catalogs.query_region(SkyCoord(ra * u.deg, dec * u.deg), radius=rad * u.deg, catalog="Gaia", version=3)
    df = t.to_pandas(); df.columns = [c.lower() for c in df.columns]
    return df[df.phot_g_mean_mag < maglim].nsmallest(2000, "phot_g_mean_mag")


def _gaia_esa(ra, dec, rad, maglim):
    """Consulta Gaia DR3 directamente en el archivo de la ESA."""
    from astroquery.gaia import Gaia
    q = ("SELECT TOP 2000 ra, dec, pmra, pmdec, phot_g_mean_mag FROM gaiadr3.gaia_source "
         f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),CIRCLE('ICRS',{ra},{dec},{rad})) "
         f"AND phot_g_mean_mag < {maglim} ORDER BY phot_g_mean_mag")
    return Gaia.launch_job(q).get_results().to_pandas()


@st.cache_data(ttl=3600, show_spinner=False)
def gaia_stars(ra, dec, rad_deg, maglim):
    """Consulta de cono en Gaia DR3. Lanza las 5 fuentes en paralelo; la primera en responder gana.
    Devuelve la tupla: (DataFrame, nombre_del_servidor)."""
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
                    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
                df = df.dropna(subset=["ra", "dec", "phot_g_mean_mag"]).astype(float)
                return df, futs[f]
            except Exception as e:
                errors.append(f"{futs[f]}: {str(e)[:80]}")
    except Exception:
        errors.append("timeout after 60 s")
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    raise RuntimeError(" | ".join(errors))

def format_hms_dms(ra_deg, dec_deg):
    """
    Convierte grados decimales (RA, Dec) a cadenas sexagesimales con formato:
    RA  -> '17h 57m 48.50s'
    Dec -> '+04° 41\' 36.1\"'
    """
    # Creamos el objeto SkyCoord a partir de los grados decimales
    c = SkyCoord(ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs")
    
    # 1. Extraemos horas (rh), minutos (rm) y segundos (rs) de la Ascensión Recta (3 valores)
    rh, rm, rs = c.ra.hms
    
    # 2. Extraemos signo (sign_val), grados (dd), minutos (dm) y segundos (ds) de la Declinación (4 valores)
    sign_val, dd, dm, ds = c.dec.signed_dms
    
    # Definimos el símbolo '+' o '-' según el valor de sign_val (+1 o -1)
    sign = "-" if sign_val < 0 else "+"
    
    # Formateamos los textos finales (puedes cambiar .2f o .1f si deseas más o menos decimales)
    ra_str = f"{int(rh):02d}h {int(rm):02d}m {rs:05.2f}s"
    dec_str = f"{sign}{int(dd):02d}° {int(dm):02d}' {ds:04.1f}\""
    
    return ra_str, dec_str


def parse_sky_input(ra_text, dec_text=None):
    """Interpreta coordenadas en formato sexagesimal (h m s / ° ' '') o decimal y retorna (ra_deg, dec_deg)."""
    s = f"{ra_text} {dec_text}".strip() if dec_text is not None else str(ra_text).strip()
    if not s:
        return None
    # Normalizamos símbolos de grados, minutos y segundos de arco para que Astropy los lea sin ambigüedad
    s_norm = s.replace("°", "d").replace("′", "m").replace("'", "m").replace("″", "s").replace('"', "s")
    # 1. Intentar par decimal simple
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)[\s,;]+([+-]?\d+(?:\.\d+)?)", s)
    if m and 0 <= float(m[1]) <= 360 and -90 <= float(m[2]) <= 90:
        return float(m[1]), float(m[2])
    # 2. Intentar sexagesimal con SkyCoord (horas/grados o grados/grados)
    if re.search(r"\d+\s*[hmsd°'\":]", s, re.IGNORECASE):
        for units in ((u.hourangle, u.deg), (u.deg, u.deg)):
            try:
                c = SkyCoord(s_norm, unit=units, frame="icrs")
                return float(c.ra.deg), float(c.dec.deg)
            except Exception:
                continue
    return None


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
        if stretch in ("raíz", "sqrt"):
            x = np.sqrt(x)
        elif stretch == "asinh":
            x = np.arcsinh(10 * x) / np.arcsinh(10)
        elif stretch == "log":
            x = np.log1p(1000 * x) / np.log1p(1000)
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
            st.download_button(f"Download {name}", gif, name, "image/gif", key=f"dl_{name}")
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

# ================================================================ TABLA DE CLASIFICACIÓN PERSISTENTE
CLS_COLS = ["key", "ra_deg", "dec_deg", "fov_side_arcsec", "moving_object_votes", "no_moving_object_votes",
            "total_votes", "last_updated_utc", "consensus"]
LAST_COL = "I"
KEY_RE = re.compile(r"^-?\d+[.,]\d+_-?\d+[.,]\d+_\d+$")
LOCAL_CSV = "classifications.csv"


def consensus(m, nm):
    return "Moving object" if m > nm else ("No moving object" if nm > m else "Tie")


@st.cache_resource
def _worksheet():
    """Conecta con Google Sheets si existen credenciales y librerías instaladas;
    si falta gspread o falla la red, retorna None y usa el archivo CSV local sin romper la app."""
    try:
        # 1. Verificamos que existan las credenciales en .streamlit/secrets.toml
        sa = dict(st.secrets["gcp_service_account"])
        gs = st.secrets["gsheet"]["sheet_id"]
        
        # 2. Importamos las librerías DENTRO del bloque try para evitar ModuleNotFoundError
        import gspread
        from google.oauth2.service_account import Credentials
        
        # 3. Autenticamos y abrimos la hoja de cálculo
        creds = Credentials.from_service_account_info(
            sa, scopes=["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        )
        ws = gspread.authorize(creds).open_by_key(gs).sheet1
        
        # 4. Aseguramos que la primera fila tenga los encabezados correctos
        first = ws.row_values(1)
        if not first:
            ws.update(values=[CLS_COLS], range_name=f"A1:{LAST_COL}1")
        elif KEY_RE.match(first[0]):
            ws.insert_row(CLS_COLS, 1)
        return ws
    except Exception:
        return None

def _clean(df):
    for c in CLS_COLS:
        if c not in df:
            df[c] = ""
    df = df[CLS_COLS].copy()
    for c in ("ra_deg", "dec_deg"):
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    for c in ("fov_side_arcsec", "moving_object_votes", "no_moving_object_votes", "total_votes"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    df["consensus"] = [consensus(a, b) for a, b in zip(df.moving_object_votes, df.no_moving_object_votes)]
    return df


def load_votes():
    ws = _worksheet()
    if ws is not None:
        vals = ws.get_all_values()
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

# ================================================================ GUÍA ASTROFÍSICA + ENLACES
# ================================================================ LIVE ASTROPHYSICAL GUIDE + LINKS
GUIA = {
    "xb": dict(icon="⚡", titulo="X-Ray Binary",
               que="Two stars locked in a tight orbit: a normal donor star and a compact object—a neutron star or black hole—stripping gas into a cosmic sinkhole. "
                   "The infalling gas spirals into an accretion disk heated to millions of degrees, emitting powerful X-rays.",
               mov="Do not expect it to shift across the background stars: it lies thousands of light-years away and its proper motion is tiny over months. What changes is its brightness: "
                   "these systems undergo outbursts and variability. Compare epochs looking for flux changes rather than positional shifts.",
               firma="Companion star continuum plus an infrared excess from the accretion disk or relativistic jet. Look for hydrogen lines (Paschen, Brackett, Brγ at 2.17 µm), "
                     "helium at 1.083 µm, and—if the donor is cool—CO bands near 2.3 µm."),
    "agn": dict(icon="🌀", titulo="Active Galactic Nucleus / Quasar",
                que="A supermassive black hole actively accreting matter at the center of a galaxy, shining brightly enough to outshine its entire host galaxy.",
                mov="Fixed against the background (at cosmological distances), but often varies in brightness over weeks or years. A flux change without displacement points to an active nucleus.",
                firma="Power-law continuum with hot-dust excess toward 3–5 µm and broad hydrogen emission lines (Hα, Paschen) redshifted according to distance."),
    "galaxia": dict(icon="🌌", titulo="Galaxy",
                    que="An island of hundreds of billions of stars, gas, and dust. In the near-infrared, dust becomes nearly transparent, revealing the true stellar structure: "
                        "the central bulge, spiral arms, and star-forming regions.",
                    mov="Stationary on human timescales: at extragalactic distances, no galaxy shifts measurably between epochs. Any localized change in the image suggests a supernova or a foreground moving source.",
                    firma="PAH (organic molecule) emission at 3.3 µm, the old-stellar-population bump near 1.6 µm, and—in star-forming galaxies—hydrogen lines such as Paschen-α (1.87 µm) and Brackett-α (4.05 µm), "
                          "redshifted by cosmic expansion."),
    "pm": dict(icon="💨", titulo="High-Proper-Motion Star",
               que="A star moving noticeably across the sky because it is very close to the Solar System or traveling at high velocity through the Galaxy.",
               mov="It shifts relative to background stars! With SPHEREx's 6.15″ pixels, stars with high proper motion (on the order of arcseconds per year, like Barnard's Star) shift appreciably over months. "
                   "Enable Gaia DR3: its cyan circles propagate with proper motion at each epoch.",
               firma="Warm stars show a smooth blackbody-like continuum. Cool dwarfs display water vapor (1.4 and 1.9 µm), CO (2.3 and 4.6 µm), and—in the coolest brown dwarfs—methane at 1.6, 2.2, and 3.3 µm."),
    "star": dict(icon="⭐", titulo="Star",
                 que="A luminous sphere of plasma powered by nuclear fusion. Its surface temperature determines its color and spectral features.",
                 mov="Nearly static against the background over a few months unless very nearby. Brightness changes between epochs indicate stellar variability (pulsating or eclipsing variables).",
                 firma="Smooth blackbody-like continuum with H absorption, CO (2.3 µm), and water bands in cool stars; circumstellar dust produces an infrared excess at 3–5 µm."),
    "sso": dict(icon="☄️", titulo="Solar System Body",
                que="An asteroid, comet, or minor planet: rocky and icy remnants from the formation of the Solar System 4.6 billion years ago.",
                mov="The main protagonist of SxMOV: it moves rapidly across the background stars. Between epochs it appears at one position and vanishes from the previous one, leaving a positive–negative pair when subtracted.",
                firma="Reflected sunlight with a red slope, hydration and water-ice absorption bands near 3 µm, and thermal emission above ~3.5 µm. Comets also show CO₂ (4.27 µm) and CO (4.67 µm) gas emission."),
    "otro": dict(icon="🔭", titulo="Astronomical Object",
                 que="SIMBAD classifies this source with a specialized catalog code. The multi-epoch method remains the same: inspect how it changes over time.",
                 mov="Compare epochs using image subtraction: anything that does not cancel out changed either in position or in brightness.",
                 firma="Inspect the 0.75–5 µm spectrum: a smooth continuum suggests a star, whereas strong emission peaks or absorption troughs indicate gas, dust, or ices."),
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
    out = [("🔗 Open in SIMBAD", simbad), ("🔗 Query NASA/IPAC IRSA", IRSA_SPHEREX_URL),
           ("🔗 Explore in Aladin Lite", aladin), ("🔗 ESA Gaia Archive", GAIA_ARCHIVE_URL)]
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
    ra_hms, dec_dms = format_hms_dms(ra, dec)
    nombre = html.escape(main_id) if main_id else f"RA {ra_hms}, Dec {dec_dms}"
    st.markdown(H(f"""<div class="guia">
        <span class="chip">LIVE ASTROPHYSICAL GUIDE</span><span class="chip">otype: {html.escape(otype) if otype else 'unclassified'}</span>
        <h3>{g['icon']} {nombre} · {g['titulo']}</h3>
        <p class="lead">{g['que']}</p>
        <div class="guia-grid">
            <div><b>EXPECTED MOTION</b>{g['mov']}</div>
            <div><b>0.75–5 µm SIGNATURES</b>{g['firma']}</div>
            <div><b>HOW TO SPOT IT IN SxMOV</b>Select a band, click <i>Search and generate</i>, and step through the epochs in the GIF and the A − B subtraction.
            {'Enable Gaia DR3 to rule out known background stars.' if key in ('pm', 'star', 'sso') else 'Compare the aperture flux across epochs.'}</div>
        </div></div>"""), unsafe_allow_html=True)
    return key, main_id

# ================================================================ VISTA 5 · EXPLORADOR
def vista_explorador():
    ss = st.session_state
    st.markdown(H("""<div class="kicker">INTERACTIVE LABORATORY · MULTI-EPOCH ANALYSIS</div>
        <div class="exp-title">SxMOV Explorer</div>"""), unsafe_allow_html=True)

    # --- Top control panel (quick search bar + 3 configuration tabs) ---
    bar = st.container(key="quickbar")
    tabs = st.tabs(["🎯 Target & Catalogs", "🖼️ Image & Bands", "⏱️ Time & Animation"])
    hud_top = st.empty()
    names = list_releases()
    ss.setdefault("rel_sel", latest(names))

    with tabs[0]:
        c1, c2, c3 = st.columns(3)
        with c1:
            how = st.radio("Target input", ["Name (SIMBAD)", "RA/Dec coordinates (h m s / ° ' '')"], key="how")
            ra_hms_in = st.text_input("Right Ascension [h m s]", key="ra_hms", disabled=how.startswith("Name"),
                                      help="Example: 17h 57m 48.498s or 08h55m10.831s (decimal degrees also accepted)")
            dec_dms_in = st.text_input("Declination [° ' '']", key="dec_dms", disabled=how.startswith("Name"),
                                       help="Example: +04° 41' 36.11\" or -07d14m42.53s (decimal degrees also accepted)")
        with c2:
            auto = st.checkbox("Latest release automatically (spherex_dr1 / qr3)", key="auto_rel")
            if auto:
                rel = latest(names)
                st.caption("Using: " + ", ".join(rel))
                deep = st.checkbox("Also include the Deep Fields (dr1, qr3)", key="deep")
                colls = tuple(rel + ([r + "_deep" for r in rel] if deep else []))
            else:
                colls = tuple(st.multiselect("Releases", names, key="rel_sel",
                                             help="Includes spherex_dr1_deep and spherex_qr3_deep (Deep Survey)."))
        with c3:
            use_gaia = st.checkbox("Compare with Gaia (visible)", key="use_gaia",
                                   help="Shows the same field from Gaia DR3 next to the SPHEREx images.")
            overlay = st.checkbox("Overlay Gaia circles on SPHEREx", key="overlay", disabled=not use_gaia)
            maglim = st.slider("Gaia limiting G magnitude", 10.0, 21.0, key="maglim", disabled=not use_gaia)

    with tabs[1]:
        c1, c2, c3 = st.columns(3)
        with c1:
            # Control idéntico a app_sin_interfaz.py: min=60, max=2000, step=30
            side = float(st.number_input("Field of view: image side [arcsec]", 60, 2000, step=30, key="side",
                                         help="Side length of the square image. Wider fields follow faster sources "
                                              "but take longer to download."))
            area = int(side ** 2)
            r_ap = st.slider("Spectrum aperture [arcsec]", 6, 40, key="r_ap")
        with c2:
            mode_lbl = st.radio("Infrared bands", ["Single band", "Sum of all bands"], key="mode_lbl")
            mode = "band" if mode_lbl.startswith("Single") else "sum"
            all_dets = st.checkbox("Download all bands (full spectra 0.75–5 µm)", key="all_dets", disabled=(mode == "sum"))
        with c3:
            stretch = st.selectbox("Stretch", ["linear", "log", "sqrt", "asinh"], key="stretch")
            neg = st.checkbox("Negative (invert intensities)", key="neg")
            bgsub = st.checkbox("Subtract background (median) per image", key="bgsub")
    with tabs[2]:
        c1, c2, c3 = st.columns(3)
        with c1:
            rng = st.date_input("Date range", key="rng")
            gap = st.slider("Gap that separates epochs [days]", 0.5, 30.0, key="gap")
        with c2:
            maxn = st.slider("Max. images to download", 5, 150, key="maxn")
        with c3:
            ms = st.slider("Time between frames [ms]", 100, 3000, step=100, key="ms")
            mark = st.checkbox("Mark center", key="mark")

    with bar:
        qa, qb, qc = st.columns([4, 2.4, 1.3])
        qa.text_input("SIMBAD name or RA/Dec", key="target", label_visibility="collapsed",
                      placeholder="SIMBAD name or RA/Dec (e.g. Barnard's Star · 08h55m10.831s -7d14m42.53s)")
        band = qb.selectbox("Spectral band", list(BANDS), key="band", label_visibility="collapsed", disabled=(mode == "sum"),
                            format_func=lambda b: f"Band {b} ({BANDS[b][0]}–{BANDS[b][1]} µm)")
        go = qc.button("🔎 Search and generate", type="primary", key="analizar")
    go = go or bool(ss.pop("_autorun", False))

    hud_top.markdown(H('<div class="top-hud">SPHEREX LATEST DATA &nbsp;·&nbsp; Active collections: '
                       f'<span class="highlight-green">{", ".join(colls) if colls else "—"}</span></div>'), unsafe_allow_html=True)

    # --- Target resolution (supports SIMBAD name, h m s / ° ' '', or decimal degrees) ---
    obj, target_str = None, ""
    if not how.startswith("Name"):
        parsed_coord = parse_sky_input(ra_hms_in, dec_dms_in)
        if parsed_coord is None:
            st.warning("Invalid RA/Dec format. Use e.g. '17h 57m 48.5s' and '+04° 41\' 36.1\"' (or decimal degrees)."); return
        ra, dec = parsed_coord
        ra_hms_str, dec_dms_str = format_hms_dms(ra, dec)
        target_str = f"{ra_hms_str} {dec_dms_str}"
        obj = identify_coords(round(ra, 5), round(dec, 5))
    else:
        q = ss["target"].strip()
        parsed_q = parse_sky_input(q)
        if parsed_q is not None:
            ra, dec = parsed_q
            obj = identify_coords(round(ra, 5), round(dec, 5))
        elif not q:
            st.info("Enter a target name (SIMBAD) or RA/Dec coordinates in the search bar."); return
        else:
            obj = resolve_name(q)
            if obj is None:
                st.warning("Object not found in SIMBAD."); return
            ra, dec = obj["ra"], obj["dec"]
        target_str = q

    ra_hms_disp, dec_dms_disp = format_hms_dms(ra, dec)
    st.markdown(
        f'<div class="coords" translate="no">RA {ra_hms_disp}&nbsp;&nbsp;Dec {dec_dms_disp} '
        f'<span style="font-size:0.95rem;color:var(--mut);">({ra:.5f}°, {dec:+.5f}°)</span></div>',
        unsafe_allow_html=True
    )

    # --- Live astrophysical guide ---
    with st.container(key="guia_" + re.sub(r"[^A-Za-z0-9]", "", f"{ra:.4f}{dec:.4f}")):
        gkey, main_id = guia_card(obj, ra, dec)
    links = link_list(ra, dec, side, main_id, gkey)
    cols = st.columns(len(links) + 1)
    for col, (lbl, url) in zip(cols, links):
        col.link_button(lbl, url)
    if cols[-1].button("🔗 Copy observation link", key="btn_share"):
        ss["_share"] = True
    if ss.get("_share"):
        st.caption("Copy this URL to share or embed this exact query:")
        st.code(share_url(target_str, band), language=None)
    if obj and obj.get("info"):
        with st.expander("SIMBAD Technical Datasheet"):
            if obj.get("desc"):
                st.write(f"**Object type:** {obj['info'].get('otype', '?')} – {obj['desc']}")
            st.dataframe(pd.DataFrame([obj["info"]]), hide_index=True)

    # --- Search & download in IRSA (SPHEREx) ---
    sig = (colls, round(ra, 5), round(dec, 5), area, mode, band, all_dets, str(rng), gap, maxn, bgsub, r_ap)
    if go:
        if not colls:
            st.error("Select at least one release."); return
        if not isinstance(rng, (tuple, list)) or len(rng) < 2:
            st.info("Select a start and an end date."); return
        d0, d1 = rng
        try:
            with st.spinner("Querying IRSA (SIA)…"):
                df = sia_search(ra, dec, colls)
        except Exception as e:
            st.error(f"⚠️ {e}  (IRSA server temporarily unavailable — HTTP 502/503.)")
            return
        df = df[(df.date.dt.date >= d0) & (df.date.dt.date <= d1)]
        dets = list(BANDS) if (mode == "sum" or all_dets) else [band]
        df = df[df.det.isin(dets)].reset_index(drop=True)
        if df.empty:
            st.error("No SPHEREx images for these parameters."); return
        if len(df) > maxn:
            df = df.iloc[np.unique(np.linspace(0, len(df) - 1, maxn).astype(int))].reset_index(drop=True)
            st.info(f"Sampled {maxn} images spread over time.")
        df["gid"] = (df.t_min.diff().fillna(0) > gap).cumsum()
        n = int(max(8, round(side / PIX)))
        w = WCS(naxis=2); w.wcs.ctype = ["RA---TAN", "DEC--TAN"]; w.wcs.crval = [ra, dec]
        w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]; w.wcs.cdelt = [-PIX / 3600, PIX / 3600]
        results, pbar = {}, st.progress(0.0, "Downloading cutouts…")
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
        st.success(f"{len(items)}/{len(df)} cutouts downloaded.")
        ss["res"] = dict(items=items, wcs=w, sig=sig, zoom=0, ra=ra, dec=dec, area=area, side=side,
                         rid=int(time.time() * 1000))

    R = ss.get("res")
    if not R:
        st.info("Set the parameters and press **Search and generate**."); return
    if R["sig"] != sig:
        st.error("⚠️ You changed parameters (field of view, band, dates, …) but the images below still come from the "
                 "previous run. Press **Search and generate** to apply them.")

    items = R["items"]
    if not items:
        st.error("Could not download any cutout. Try again or adjust the parameters."); return

    with st.container(key=f"res_{R['rid']}"):
        frames = make_frames(items, mode, band, bgsub)
        labels = [f["label"] for f in frames]
        n_pix = frames[0]["arr"].shape[0]

        # --- Dynamic Gaia DR3 query (5 parallel mirrors) ---
        gaia = None
        if use_gaia:
            try:
                with st.spinner("Querying Gaia DR3…"):
                    gaia, gaia_src = gaia_stars(round(R["ra"], 5), round(R["dec"], 5), R["side"] * 0.75 / 3600, float(maglim))
                st.caption(f"Gaia DR3 source used: **{gaia_src}** ({len(gaia)} stars found)")
                if len(gaia) == 0:
                    st.info("Gaia returned no sources in this field for the chosen magnitude limit.")
            except Exception as e:
                st.warning(f"Gaia is not available right now: {e}")
        has_gaia = gaia is not None and len(gaia) > 0

        # ---- Photometric images (SPHEREx | Gaia DR3) ----
        st.header("Photometric images (FITS cutouts)")
        zc = st.columns([1, 1, 6])
        if zc[0].button("🔍 +"):
            R["zoom"] = min(len(ZOOMS) - 1, R["zoom"] + 1)
        if zc[1].button("🔍 −"):
            R["zoom"] = max(0, R["zoom"] - 1)
        zoom = ZOOMS[R["zoom"]]
        zc[2].caption(f"Zoom ×{zoom} · generated field: {R['side']:.0f}″ per side ({R['area']:,} arcsec²) · {len(frames)} epochs")
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

        # ---- Subtraction A−B (Método exacto de app_sin_interfaz.py) ----
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
        st.caption("Fixed stars cancel out; what remains (positive in one epoch and negative in the other, "
                   "like a pair) suggests a moving object. Use the same band in both epochs.")
        # ---- Difference imaging (all epochs) ----
        st.subheader("Difference imaging (all epochs)")
        if len(frames) < 2:
            st.info("At least two epochs are needed.")
        else:
            dc1, dc2, dc3 = st.columns(3)
            refmode = dc1.selectbox("Reference built from the other epochs", ["median", "mean"],
                                    help="Median is robust: a moving object barely contaminates the reference.")
            ksigma = dc2.slider("Show only above N × noise (σ)", 1.0, 10.0, 3.0, 0.5)
            dshow = dc3.selectbox("Display", ["Positive and negative", "Positive only"])
            dframes = []
            for i, f in enumerate(frames):
                others = np.stack([g["arr"] for j, g in enumerate(frames) if j != i])
                refim = np.nanmedian(others, axis=0) if refmode == "median" else np.nanmean(others, axis=0)
                d = f["arr"] - refim
                sig_i = 1.4826 * float(np.nanmedian(np.abs(d - np.nanmedian(d)))) + 1e-12  # robust noise (MAD)
                dframes.append(dict(f, arr=np.where(np.abs(d) > ksigma * sig_i, d, 0.0), sig=sig_i))
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
                return draw_extras(im, f"Difference {f['label']} (>{ksigma:g}σ)", mark, (x1 - x0) * PIX)

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

        # ---- Spectra ----
        st.header("Spectra per epoch")
        gids = [f["gid"] for f in frames]
        sg = spectrum_gif(items, gids, labels, ms)
        if sg:
            st.image(sg)
            if mode == "band" and not all_dets:
                st.caption("With a single band the spectrum covers a narrow range; enable «Download all bands» for 0.75–5 µm.")
            ea, eb = st.columns(2)
            sa = ea.selectbox("Spectrum epoch A", range(len(frames)), format_func=lambda i: labels[i], key="sa")
            sb_ = eb.selectbox("Spectrum epoch B", range(len(frames)), format_func=lambda i: labels[i], key="sb", index=len(frames) - 1)
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

        # ---- Metadata + Legend ----
        st.header("Image metadata (FITS headers)")
        e = st.selectbox("Epoch", range(len(frames)), format_func=lambda i: labels[i], key="meta")
        rows = []
        for i in items:
            if i["gid"] == gids[e]:
                rows.append({"detector": i["det"], "center λ [µm]": round(i["wave"], 4), "flux [mJy]": round(i["flux"], 3), **i["meta"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True)
        with st.expander("📖 Column legend & contamination flags"):
            st.markdown("""
| Column | Meaning |
|---|---|
| `detector` | SPHEREx detector array (1–6); each one covers one infrared band. |
| `center λ [µm]` | Wavelength at the target pixel, from the spectral WCS (varies across the detector). |
| `flux [mJy]` | Background-subtracted aperture flux at the target (aperture set above; background from an annulus). |
| `DETECTOR`, `OBSID` | Detector and observation identifiers from the FITS header. |
| `DATE` | Date recorded in the FITS header. |
| `PSF_FWHM` | Full width at half maximum of the point spread function. |
| `VERSION` | Processing version from the primary header. |
| `N_*` | Pixel counters from the L2 processing: number of pixels carrying each flag; keywords ending in `_NB` exclude pixels also flagged bad and those ending in `_NS` exclude pixels also flagged SOURCE. |
| `zodi_mediana` | Median of the zodiacal-light model in the cutout [MJy/sr]; it is **not** subtracted from the IMAGE. |
| `img_mediana` | Median surface brightness of the IMAGE cutout [MJy/sr]. |
| `frac_NaN` | Fraction of cutout pixels with no data (0 = complete, 1 = empty). |
""")

        # ---- Field classification ----
        st.header("Classify this field")
        st.write("Did you see an object moving relative to the background stars in this field of view?")
        key_now = f"{R['ra']:.{COORD_DECIMALS}f}_{R['dec']:.{COORD_DECIMALS}f}_{int(R['side'])}"
        voted = ss.setdefault("voted", set())
        b1, b2, _ = st.columns([1.2, 1.2, 3])
        choice = None
        if b1.button("Moving object present", type="primary", key="vote_yes"):
            choice = True
        if b2.button("No moving object", key="vote_no"):
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
        st.download_button("Download table (CSV)", votes.to_csv(index=False), "classifications.csv", "text/csv", key="dl_csv_votes")


# ================================================================ DISPATCHER
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
    st.markdown(footer_html(55, "NASA International Space Apps Challenge — Bogotá Hub"), unsafe_allow_html=True)