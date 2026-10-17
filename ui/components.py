"""Composants d'interface réutilisables (aucune logique scientifique ici)."""
import html
import re
from contextlib import contextmanager

import streamlit as st

DISCLAIMER = ("Outil de recherche. Les prédictions du modèle sont des estimations algorithmiques : "
              "elles ne constituent pas un diagnostic médical.")

# Logo : un motif pointe-onde (la signature d'une crise d'absence) tracé en une ligne
LOGO_SVG = ('<svg width="40" height="28" viewBox="0 0 40 28" fill="none" aria-hidden="true">'
            '<rect x="0" y="0" width="40" height="28" rx="6" fill="#1D4E96"/>'
            '<polyline points="4,15 11,15 14,5 17,24 20,15 26,9 30,15 36,15" stroke="#FFFFFF" stroke-width="2" '
            'stroke-linejoin="round" stroke-linecap="round" fill="none"/></svg>')


def md(t):
    """Mini-balisage sûr : échappe le HTML puis gère **gras**, `code` et les retours à la ligne."""
    t = html.escape(str(t), quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
    return t.replace("\n", "<br>")


def _html(s):
    st.markdown(s, unsafe_allow_html=True)


# ----------------------------------------------------------------------------- barre latérale
def brand(title="EEG · Absence", subtitle="Détection de crises"):
    _html(f'<div class="sb-brand">{LOGO_SVG}<div><div class="t1">{html.escape(title)}</div>'
          f'<div class="t2">{html.escape(subtitle)}</div></div></div>')


def sidebar_label(text):
    _html(f'<div class="sb-label">{html.escape(text)}</div>')


def facts(items):
    """Liste clé / valeur compacte (valeurs en police mono)."""
    rows = "".join(f'<div class="f"><span class="k">{html.escape(k)}</span><span class="v">{html.escape(str(v))}</span></div>' for k, v in items)
    _html(f'<div class="sb-facts">{rows}</div>')


def sidebar_note(text):
    _html(f'<div class="sb-note">{html.escape(text)}</div>')


# ----------------------------------------------------------------------------- navigation
def stepper(labels, current, visited, on_select, helps=None):
    """Navigation principale en étapes : la page courante est soulignée, les étapes déjà consultées sont cochées."""
    _html('<div class="stepper-anchor"></div>')
    for i, c in enumerate(st.columns(len(labels))):
        mark = " ✓" if (i in visited and i != current and i != 0) else ""
        text = (labels[i] if i == 0 else f"{i} {labels[i]}") + mark
        c.button(text, key=f"step_btn_{i}", type="primary" if i == current else "secondary",
                 on_click=on_select, args=(i,), help=(helps[i] if helps else None))


def step_nav(i, labels, on_select):
    """Pied de page : retour à l'étape précédente / passage à la suivante."""
    _html('<div class="stepnav-anchor"></div>')
    left, right = st.columns(2)
    if i > 0:
        left.button(f"← Précédente : {labels[i - 1]}", key=f"prev_{i}", on_click=on_select, args=(i - 1,))
    if i < len(labels) - 1:
        right.button(f"Suivante : {labels[i + 1]} →", key=f"next_{i}", type="primary", on_click=on_select, args=(i + 1,))


def page_header(step, total, title, lead, ctx):
    items = "".join(f'<span class="ctx-item"><span class="ctx-k">{html.escape(k)}</span>'
                    f'<span class="ctx-v">{html.escape(str(v))}</span></span>' for k, v in ctx)
    eyebrow = "Projet Data Mining · 3AINFO" if step == 0 else f"Étape {step} sur {total} · pipeline d'analyse"
    _html(f'<div class="ph"><div class="ph-eyebrow">{eyebrow}</div><h1 class="ph-title">{html.escape(title)}</h1>'
          f'<p class="ph-lead">{md(lead)}</p><div class="ph-ctx">{items}</div></div>')


# ----------------------------------------------------------------------------- blocs de contenu
def section(title, desc=None):
    d = f'<div class="sec-d">{md(desc)}</div>' if desc else ""
    _html(f'<div class="sec"><div class="sec-t">{html.escape(title)}</div>{d}</div>')


@contextmanager
def panel(title=None):
    """Panneau de paramètres (un seul niveau d'encadré dans toute l'application)."""
    _html('<div class="panel-anchor"></div>')
    with st.container(border=True):
        if title:
            _html(f'<div class="panel-title">{html.escape(title)}</div>')
        yield


def callout(kind, body, title=None):
    """kind : info | warn | critical | ok | neutral. Le titre rend le sens lisible sans la couleur."""
    kind = {"info": ""}.get(kind, f" callout-{kind}")
    t = f'<div class="ct">{html.escape(title)}</div>' if title else ""
    _html(f'<div class="callout{kind}">{t}<p>{md(body)}</p></div>')


def chips(items):
    _html('<div class="chips">' + "".join(f'<span class="chip">{html.escape(str(i))}</span>' for i in items) + "</div>")


def legend(items):
    """items : (libellé, couleur, style) avec style fill | line | outline. Chaque entrée porte un libellé explicite."""
    out = []
    for label, color, style in items:
        css = {"fill": f"background:{color}", "line": f"border-color:{color}", "outline": f"border-color:{color}"}[style]
        cls = {"fill": "", "line": " line", "outline": " outline"}[style]
        out.append(f'<span class="lg"><i class="{cls.strip()}" style="{css}"></i>{html.escape(label)}</span>')
    _html('<div class="legend">' + "".join(out) + "</div>")


def badge(text, kind="neutral"):
    return f'<span class="badge badge-{kind}">{html.escape(text)}</span>'


def verdict(kind, title, detail):
    """Verdict d'une fenêtre : mot + couleur (jamais la couleur seule)."""
    _html(f'<div class="verdict">{badge(title, kind)}<small>{html.escape(detail)}</small></div>')


def flow(steps):
    """Chaîne de traitement : liste de (titre, description)."""
    cells = "".join(f'<div><div class="n">{i + 1:02d}</div><div class="h">{html.escape(h)}</div>'
                    f'<div class="d">{html.escape(d)}</div></div>' for i, (h, d) in enumerate(steps))
    _html(f'<div class="flow">{cells}</div>')


def status_table(rows):
    """rows : (nom, fait ?, texte). L'état est écrit en toutes lettres."""
    body = "".join(
        f'<tr><td>{html.escape(n)}</td><td><span class="tag {"tag-done" if ok else "tag-todo"}">{"Fait" if ok else "À faire"}</span>'
        f'{md(t)}</td></tr>' for n, ok, t in rows)
    _html(f'<table class="status">{body}</table>')


def disclaimer():
    _html(f'<div class="disclaimer">{html.escape(DISCLAIMER)}</div>')
