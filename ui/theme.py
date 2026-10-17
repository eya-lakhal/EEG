"""Identité visuelle de l'application : palette, typographie, CSS et gabarit Plotly.

Une seule source pour les couleurs : tout le reste (CSS, graphiques, composants) en dérive.
Inspiration : cerveau en camaïeu bleu marine sur fond rose, équipement hospitalier, publications de neurosciences.
Palette : fond rose bébé, texte noir et bleu marine (sobre, lisible, usage clinique).
"""
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ----------------------------------------------------------------------------- palette
IVORY = "#FBE4EA"          # fond de page (rose bébé)
SURFACE = "#FFF6F8"        # surfaces claires (panneaux, zone de tracé) : rose très pâle, presque blanc
IVORY_DEEP = "#F6D2DD"     # barre latérale
LINE = "#E9B9C7"           # bordures
LINE_SOFT = "#F2D0DA"      # grilles des graphiques

INK = "#14171C"            # texte principal (noir)
SLATE = "#34476B"          # texte secondaire (bleu ardoise ; contraste > 7:1 sur le rose)
SLATE_LIGHT = "#6B7C9A"

PETROL_900 = "#0A2447"     # bleu marine profond : structure, titres (nom historique « petrol », la teinte est maintenant bleue)
PETROL = "#1D4E96"         # bleu médical : action principale, navigation active
PETROL_500 = "#3A74C0"
PETROL_200 = "#C9DBF2"

SAGE = "#7FA593"           # vert sauge : éléments secondaires, rassurants
SAGE_DARK = "#3F6B5B"
SAGE_LIGHT = "#DCEBE3"

COPPER = "#B5457A"         # rose soutenu : filets, repères, fonds (nom historique « copper »)
COPPER_TEXT = "#8E2E5E"    # rose foncé pour le texte
COPPER_LIGHT = "#F2C3D5"

CORAL = "#C0392B"          # rouge clinique : événements critiques (crises), distinct du fond rose
CORAL_DARK = "#8A1F16"     # rouge foncé pour le texte et les badges
CORAL_LIGHT = "#F8CFCB"

AMBER = "#D8A23E"          # ambre : phase pré-critique
AMBER_LIGHT = "#F6E3B4"
PLUM = "#86699A"           # prune : phase post-critique (réservée, non annotée dans les données)
INTERICTAL = "#8FA6C0"     # gris-bleu : phase inter-critique (fond)

# Rôles sémantiques dans les graphiques (une couleur = une signification, partout)
ROLE = {
    "signal": "#1B2433",       # tracé EEG
    "raw": "#9AA3AA",          # signal brut
    "filtered": PETROL_500,    # signal filtré
    "normalized": SAGE_DARK,   # signal normalisé
    "reference": CORAL,        # annotation de référence (experts)
    "prediction": PETROL,      # prédiction du modèle
    "proba": COPPER,           # probabilité de crise
    "window": COPPER,          # fenêtre courante
    "preictal": AMBER,
    "ictal": CORAL,
    "postictal": PLUM,
    "interictal": INTERICTAL,
    "normal": INTERICTAL,
}
# Phases réellement présentes dans les données (la phase post-critique n'est pas annotée)
PHASES = [
    ("interictal", "Inter-critique (fond)"),
    ("preictal", "Pré-critique (zone avant le début annoté)"),
    ("ictal", "Critique (annotation de référence)"),
]

FONT_UI = "'IBM Plex Sans','Segoe UI',system-ui,-apple-system,sans-serif"
FONT_SERIF = "'Newsreader',Georgia,'Times New Roman',serif"
FONT_MONO = "'IBM Plex Mono',Consolas,'Courier New',monospace"

# Couleurs de séries génériques (courbes de features) : aucune ne reprend une couleur de phase ou d'annotation
SERIES = [PETROL_500, SAGE_DARK, "#8A7F6A", SLATE, PETROL_900, COPPER]

# échelles de couleur perceptuellement ordonnées (luminance monotone)
SEQ = ["#0A2447", "#1D4E96", "#3A74C0", "#8DB4E2", "#CFE0F4", "#FFF1F5"]
DIV = [[0.0, "#1D4E96"], [0.25, "#8DB4E2"], [0.5, "#FFF6F8"], [0.75, "#E79A9A"], [1.0, "#A83232"]]


# ----------------------------------------------------------------------------- gabarit Plotly
def _axis():
    return dict(gridcolor=LINE_SOFT, linecolor=LINE, zerolinecolor=LINE, ticks="outside", tickcolor=LINE,
                tickfont=dict(size=11, color=SLATE), title_font=dict(size=12, color=SLATE), automargin=True)


def register_plotly_template():
    pio.templates["eeg"] = go.layout.Template(layout=dict(
        font=dict(family=FONT_UI, size=12, color=INK),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor=SURFACE,
        colorway=[PETROL_500, COPPER, SAGE_DARK, CORAL, PLUM, SLATE_LIGHT, AMBER],
        xaxis=_axis(), yaxis=_axis(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)",
                    font=dict(size=11, color=SLATE)),
        margin=dict(l=56, r=16, t=48, b=44),
        title=dict(font=dict(family=FONT_UI, size=13, color=SLATE), x=0, xanchor="left"),
        hoverlabel=dict(bgcolor=INK, font=dict(family=FONT_UI, color=IVORY, size=12), bordercolor=INK),
        coloraxis=dict(colorbar=dict(outlinewidth=0, ticks="outside", tickcolor=LINE, tickfont=dict(size=10, color=SLATE))),
    ))
    pio.templates.default = "plotly+eeg"


# ----------------------------------------------------------------------------- CSS
CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');

:root {{
  --ivory:{IVORY}; --surface:{SURFACE}; --ivory-deep:{IVORY_DEEP}; --line:{LINE}; --line-soft:{LINE_SOFT};
  --ink:{INK}; --slate:{SLATE}; --slate-light:{SLATE_LIGHT};
  --petrol-900:{PETROL_900}; --petrol:{PETROL}; --petrol-500:{PETROL_500}; --petrol-200:{PETROL_200};
  --sage:{SAGE}; --sage-dark:{SAGE_DARK}; --sage-light:{SAGE_LIGHT};
  --copper:{COPPER}; --copper-text:{COPPER_TEXT}; --copper-light:{COPPER_LIGHT};
  --coral:{CORAL}; --coral-dark:{CORAL_DARK}; --coral-light:{CORAL_LIGHT};
  --amber:{AMBER}; --amber-light:{AMBER_LIGHT};
  --font-ui:{FONT_UI}; --font-serif:{FONT_SERIF}; --font-mono:{FONT_MONO};
}}

/* ---------- base typographique ---------- */
html, body, .stApp, .stMarkdown, [data-testid="stMarkdownContainer"] p, [data-testid="stWidgetLabel"] p,
label, button, input, textarea, select, [data-baseweb="select"] *, [data-baseweb="tab"], [data-baseweb="radio"] *,
[data-testid="stMetricLabel"], [data-testid="stCaptionContainer"], [data-testid="stExpander"] summary {{
  font-family: var(--font-ui) !important;
}}
.stApp {{
  background-color: var(--ivory);
  /* Couche d'arrière-plan (jamais le conteneur : textes et graphiques restent 100 % opaques) :
     1) un voile rose qui règle la visibilité du cerveau (0.74 = cerveau visible à ~26 % ; monter la valeur pour le faire pâlir),
     2) l'image static/bg_brain.png (cerveau en camaïeu bleu marine, bords déjà fondus dans le rose). */
  background-image: linear-gradient(rgba(251,228,234,.74), rgba(251,228,234,.74)), url("/app/static/bg_brain.png");
  background-repeat: no-repeat, no-repeat;
  background-size: cover, min(max(84vw, 760px), 1500px) auto;
  background-position: center, right 2vw center;
  background-attachment: scroll, fixed;
  color: var(--ink); font-size: 15px; line-height: 1.55;
}}
.stApp [data-testid="stMarkdownContainer"] p {{ font-size: 15px; line-height: 1.6; }}
code, [data-testid="stMarkdownContainer"] code {{
  font-family: var(--font-mono) !important; font-size: .86em; background: var(--ivory-deep); color: var(--petrol-900);
  padding: .08em .35em; border-radius: 3px; border: 0;
}}

/* ---------- chrome Streamlit ---------- */
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stHeader"] {{ background: transparent; }}
footer {{ visibility: hidden; }}
.block-container, [data-testid="stAppViewBlockContainer"] {{
  max-width: 1260px; padding: 1.6rem 2.6rem 3.5rem 2.6rem;
}}
@media (max-width: 900px) {{ .block-container, [data-testid="stAppViewBlockContainer"] {{ padding: 1rem 1rem 3rem 1rem; }} }}

/* ---------- barre latérale ---------- */
[data-testid="stSidebar"] {{ background: var(--ivory-deep); border-right: 1px solid var(--line); }}
[data-testid="stSidebarUserContent"] {{ padding-top: .4rem; }}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{ font-size: 13px; color: var(--ink); font-weight: 500; }}
.sb-brand {{ display:flex; align-items:center; gap:.7rem; padding:.3rem 0 1rem 0; border-bottom:1px solid var(--line); margin-bottom:1rem; }}
.sb-brand svg {{ flex:none; }}
.sb-brand .t1 {{ font-family: var(--font-serif); font-size: 1.28rem; line-height:1.15; color: var(--petrol-900); font-weight:500; }}
.sb-brand .t2 {{ font-size: .72rem; letter-spacing:.09em; text-transform:uppercase; color: var(--slate); margin-top:.15rem; }}
.sb-label {{ font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; color: var(--slate); font-weight:600; margin:1.1rem 0 .35rem 0; }}
.sb-facts {{ margin:.5rem 0 .2rem 0; }}
.sb-facts .f {{ display:flex; justify-content:space-between; align-items:baseline; padding:.28rem 0; border-bottom:1px solid var(--line-soft); font-size:.82rem; }}
.sb-facts .k {{ color: var(--slate); }}
.sb-facts .v {{ font-family: var(--font-mono); color: var(--ink); }}
.sb-note {{ font-size:.75rem; line-height:1.45; color: var(--slate); border-top:1px solid var(--line); padding-top:.8rem; margin-top:1.4rem; }}

/* ---------- navigation par étapes (stepper) ---------- */
.element-container:has(.stepper-anchor) {{ display:none; }}
.element-container:has(.stepper-anchor) + [data-testid="stHorizontalBlock"] {{
  gap: 0 !important; flex-wrap: nowrap !important; justify-content: space-between; overflow-x: auto; border-bottom: 1px solid var(--line); margin-bottom: 1.6rem;
}}
.element-container:has(.stepper-anchor) + [data-testid="stHorizontalBlock"] [data-testid="column"] {{
  min-width: 0 !important; flex: 0 0 auto !important; width: auto !important;
}}
.element-container:has(.stepper-anchor) + [data-testid="stHorizontalBlock"] button {{
  width: auto; border: 0 !important; border-radius: 0 !important; background: transparent !important; box-shadow:none !important;
  border-bottom: 3px solid transparent !important; padding: .72rem .8rem .6rem .8rem; min-height: 0;
  color: var(--slate) !important; font-size: 13.5px; font-weight: 500;
}}
.element-container:has(.stepper-anchor) + [data-testid="stHorizontalBlock"] button:hover {{ color: var(--petrol-900) !important; background: rgba(29,78,150,.08) !important; }}
.element-container:has(.stepper-anchor) + [data-testid="stHorizontalBlock"] button[kind="primary"],
.element-container:has(.stepper-anchor) + [data-testid="stHorizontalBlock"] button[data-testid="baseButton-primary"] {{
  color: var(--petrol-900) !important; font-weight: 600; border-bottom: 3px solid var(--copper) !important; background: transparent !important;
}}
.element-container:has(.stepper-anchor) + [data-testid="stHorizontalBlock"] button p {{ white-space: nowrap; }}

/* ---------- en-tête de page ---------- */
.ph {{ margin: 0 0 1.4rem 0; }}
.ph-eyebrow {{ font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color: var(--copper-text); font-weight:600; }}
.ph-title {{ font-family: var(--font-serif); font-size: 2.35rem; line-height:1.12; font-weight:500; color: var(--petrol-900);
  letter-spacing:-.012em; margin: .25rem 0 .45rem 0; padding:0; }}
.ph-lead {{ font-size: 1.02rem; color: var(--slate); max-width: 62rem; margin: 0 0 .9rem 0; line-height:1.55; }}
.ph-ctx {{ display:flex; flex-wrap:wrap; gap:.4rem 1.6rem; padding:.55rem 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }}
.ctx-item {{ display:inline-flex; gap:.5rem; align-items:baseline; font-size:.83rem; }}
.ctx-k {{ color: var(--slate); text-transform:uppercase; letter-spacing:.07em; font-size:.68rem; font-weight:600; }}
.ctx-v {{ font-family: var(--font-mono); color: var(--ink); }}

/* ---------- sections ---------- */
.sec {{ margin: 2rem 0 .8rem 0; }}
.sec-t {{ font-size:.74rem; letter-spacing:.11em; text-transform:uppercase; font-weight:600; color: var(--petrol); display:flex; align-items:center; gap:.7rem; }}
.sec-t::after {{ content:""; flex:1; height:1px; background: var(--line); }}
.sec-d {{ font-size:.9rem; color: var(--slate); margin-top:.35rem; max-width: 60rem; }}

/* ---------- panneau de paramètres ---------- */
.element-container:has(.panel-anchor) {{ display:none; }}
.element-container:has(.panel-anchor) + [data-testid="stVerticalBlockBorderWrapper"] {{
  background: rgba(255,246,248,.93); border: 1px solid var(--line) !important; border-radius: 6px; padding: .35rem .6rem .15rem .6rem;
}}
.panel-title {{ font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; font-weight:600; color: var(--slate); margin:.2rem 0 .1rem 0; }}

/* ---------- messages ---------- */
.callout {{ border-left: 3px solid var(--petrol-500); background: rgba(29,78,150,.07); padding:.7rem 1rem; margin:.7rem 0 1rem 0; border-radius: 0 4px 4px 0; font-size:.92rem; }}
.callout .ct {{ font-weight:600; font-size:.74rem; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.15rem; color: var(--petrol); }}
.callout p {{ margin:0; }}
.callout-warn {{ border-left-color: var(--copper); background: rgba(181,69,122,.10); }}
.callout-warn .ct {{ color: var(--copper-text); }}
.callout-critical {{ border-left-color: var(--coral); background: rgba(192,57,43,.09); }}
.callout-critical .ct {{ color: var(--coral-dark); }}
.callout-ok {{ border-left-color: var(--sage-dark); background: rgba(127,165,147,.14); }}
.callout-ok .ct {{ color: var(--sage-dark); }}
.callout-neutral {{ border-left-color: var(--slate-light); background: rgba(52,71,107,.07); }}
.callout-neutral .ct {{ color: var(--slate); }}

/* ---------- étiquettes, légendes, badges ---------- */
.chips {{ display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0 .6rem 0; }}
.chip {{ font-family: var(--font-mono); font-size:.8rem; background: var(--sage-light); color: var(--petrol-900); padding:.18rem .55rem; border-radius:3px; border:1px solid rgba(63,107,91,.30); }}
.legend {{ display:flex; flex-wrap:wrap; gap:.4rem 1.3rem; font-size:.8rem; color: var(--slate); margin:.25rem 0 .6rem 0; }}
.lg {{ display:inline-flex; align-items:center; gap:.45rem; }}
.lg i {{ display:inline-block; width:1.5rem; height:.7rem; border-radius:2px; }}
.lg i.line {{ height:0; border-top:2px solid; border-radius:0; }}
.lg i.outline {{ background: transparent !important; border: 2px solid; }}
.badge {{ display:inline-block; font-size:.74rem; letter-spacing:.07em; text-transform:uppercase; font-weight:600; padding:.22rem .65rem; border-radius:3px; border:1px solid; }}
.badge-ok {{ color: var(--petrol-900); background: var(--petrol-200); border-color: var(--petrol-500); }}
.badge-neutral {{ color: var(--slate); background: var(--ivory-deep); border-color: var(--line); }}
.badge-warn {{ color: #6B4510; background: var(--amber-light); border-color: var(--amber); }}
.badge-critical {{ color: var(--coral-dark); background: var(--coral-light); border-color: var(--coral); }}
.verdict {{ display:flex; flex-direction:column; align-items:flex-start; gap:.3rem; }}
.verdict small {{ color: var(--slate); font-size:.8rem; }}

/* ---------- état de l'analyse / étapes ---------- */
.flow {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap:0; border:1px solid var(--line); border-radius:6px; background: rgba(255,246,248,.93); overflow:hidden; }}
.flow > div {{ padding:.8rem .9rem; border-right:1px solid var(--line); font-size:.85rem; }}
.flow > div:last-child {{ border-right:0; }}
.flow .n {{ font-family: var(--font-mono); font-size:.72rem; color: var(--copper-text); }}
.flow .h {{ font-weight:600; color: var(--petrol-900); margin:.1rem 0 .25rem 0; }}
.flow .d {{ color: var(--slate); font-size:.8rem; line-height:1.4; }}
.status, .status tbody, .status tr {{ border:0 !important; background:transparent !important; }}
.status {{ width:100%; border-collapse:collapse; font-size:.88rem; }}
.status td {{ padding:.5rem .4rem; border:0 !important; border-bottom:1px solid var(--line-soft) !important; vertical-align:top; background:transparent !important; }}
.status td:first-child {{ width: 9.5rem; font-weight:600; color: var(--petrol-900); }}
.status .tag {{ white-space:nowrap; display:inline-block; font-size:.68rem; letter-spacing:.07em; text-transform:uppercase; font-weight:600; padding:.1rem .45rem; border-radius:3px; margin-right:.5rem; border:1px solid; }}
.tag-done {{ color: var(--sage-dark); border-color: var(--sage); background: rgba(127,165,147,.14); }}
.tag-todo {{ color: var(--slate); border-color: var(--line); background: transparent; }}

/* ---------- métriques ---------- */
[data-testid="stMetric"] {{ border-left: 2px solid var(--copper-light); padding: .15rem 0 .15rem .85rem; }}
[data-testid="stMetricLabel"] p {{ font-size:.7rem !important; letter-spacing:.09em; text-transform:uppercase; color: var(--slate) !important; font-weight:600; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{ font-family: var(--font-mono) !important; font-size: 1.55rem !important; color: var(--petrol-900) !important; font-weight:500; }}

/* ---------- onglets ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 1.6rem; border-bottom: 1px solid var(--line); }}
.stTabs [data-baseweb="tab"] {{ padding: .55rem 0 .5rem 0; background: transparent; color: var(--slate); font-weight:500; font-size:.92rem; }}
.stTabs [data-baseweb="tab"][aria-selected="true"] {{ color: var(--petrol-900); font-weight:600; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: var(--copper); height: 2px; }}
.stTabs [data-baseweb="tab-border"] {{ background-color: transparent; }}

/* ---------- boutons ---------- */
.stButton > button, .stDownloadButton > button {{ border-radius: 4px; font-weight:500; font-size:.9rem; padding:.42rem 1rem; transition: background .15s, border-color .15s; }}
.stButton > button[kind="secondary"], .stButton > button[data-testid="baseButton-secondary"] {{
  background: var(--surface); border: 1px solid var(--line); color: var(--petrol-900);
}}
.stButton > button[kind="secondary"]:hover, .stButton > button[data-testid="baseButton-secondary"]:hover {{ border-color: var(--petrol); color: var(--petrol); background: #fff; }}
.stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {{ background: var(--petrol); border: 1px solid var(--petrol); color: #fff; }}
.stButton > button[kind="primary"]:hover, .stButton > button[data-testid="baseButton-primary"]:hover {{ background: var(--petrol-900); border-color: var(--petrol-900); color:#fff; }}
.stButton > button:focus-visible {{ outline: 2px solid var(--copper); outline-offset: 2px; }}

/* ---------- saisie, curseurs ---------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="base-input"] {{ border-radius: 4px !important; border-color: var(--line) !important; background: var(--surface) !important; }}
[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background: var(--surface) !important; }}
[data-testid="stSlider"] [data-testid="stThumbValue"] {{ font-family: var(--font-mono) !important; font-size:.78rem; color: var(--petrol-900); }}
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] {{ font-family: var(--font-mono) !important; font-size:.72rem; color: var(--slate); }}
[data-baseweb="tag"] {{ background: var(--sage-light) !important; color: var(--petrol-900) !important; border-radius:3px !important; }}
[data-testid="stCaptionContainer"] {{ color: var(--slate); font-size:.82rem; }}

/* ---------- tableaux, expanders ---------- */
[data-testid="stDataFrame"] {{ border: 1px solid var(--line); border-radius: 4px; background: var(--surface); }}
[data-testid="stExpander"] {{ border: 1px solid var(--line); border-radius: 6px; background: rgba(255,246,248,.93); }}
[data-testid="stExpander"] summary {{ font-weight:500; color: var(--petrol-900); }}
hr {{ border-color: var(--line); }}

/* ---------- navigation entre étapes (pied de page) ---------- */
.element-container:has(.stepnav-anchor) {{ border-top:1px solid var(--line); margin-top:2.4rem; padding-top:.2rem; }}
.element-container:has(.stepnav-anchor) + [data-testid="stHorizontalBlock"] [data-testid="column"]:last-child {{ display:flex; justify-content:flex-end; }}
.disclaimer {{ font-size:.78rem; color: var(--slate); margin-top:1rem; }}

/* ---------- écrans moyens et petits : les colonnes passent à la ligne au lieu de s'écraser ---------- */
@media (max-width: 1100px) {{
  [data-testid="stHorizontalBlock"] {{ flex-wrap: wrap; row-gap: .6rem; }}
  [data-testid="stHorizontalBlock"] > [data-testid="column"] {{ min-width: 260px !important; flex: 1 1 260px !important; }}
  .ph-title {{ font-size: 1.9rem; }}
}}

/* ---------- accessibilité ---------- */
a:focus-visible, button:focus-visible, [role="tab"]:focus-visible {{ outline: 2px solid var(--copper); outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
"""


def apply_theme():
    """À appeler une fois, tout en haut de l'application (après set_page_config)."""
    register_plotly_template()
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
