"""Shared Niffler theme: colors, CSS injection, and reusable themed
components used across every Streamlit page."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

GOLD = "#cfa63d"
GOLD_BRIGHT = "#e8c766"
BRONZE = "#8a6d3b"
EMERALD = "#5aa17f"
MAROON = "#c96b6b"
PARCHMENT = "#eee1c2"
PARCHMENT_DIM = "#b9ab86"
PANEL = "#161309"
PANEL_2 = "#1d1810"
HAIRLINE = "#3a331f"


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@500;600&family=Cormorant+Garamond:ital,wght@0,400;0,600;1,400&display=swap');

        h1, h2, h3 {{
            font-family: 'Cinzel', serif !important;
            letter-spacing: 1px;
        }}

        .niffler-wordmark {{
            font-family: 'Cinzel', serif;
            font-weight: 600;
            font-size: 38px;
            text-align: center;
            letter-spacing: 4px;
            background: linear-gradient(135deg, {GOLD_BRIGHT}, {GOLD} 55%, {BRONZE});
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin-bottom: 0;
        }}
        .niffler-tagline {{
            text-align: center;
            font-style: italic;
            color: {PARCHMENT_DIM};
            font-size: 16px;
            margin-top: 4px;
            margin-bottom: 28px;
        }}

        .niffler-card {{
            background: {PANEL};
            border: 1px solid {HAIRLINE};
            border-radius: 10px;
            padding: 16px 18px;
            margin-bottom: 10px;
        }}
        .niffler-card .ticker {{
            font-weight: 600;
            font-size: 16px;
            color: {PARCHMENT};
        }}
        .niffler-card .price {{
            font-size: 12px;
            color: {PARCHMENT_DIM};
        }}
        .niffler-card .pct.up {{ color: {GOLD_BRIGHT}; }}
        .niffler-card .pct.down {{ color: {MAROON}; }}
        .niffler-card .pct {{
            font-size: 22px;
            font-weight: 600;
            margin: 8px 0 2px;
        }}
        .niffler-card .rationale {{
            font-family: 'Cormorant Garamond', serif;
            font-style: italic;
            font-size: 14px;
            color: {PARCHMENT_DIM};
            margin-top: 6px;
        }}

        .niffler-whisper {{
            background: {PANEL_2};
            border-left: 2px solid {GOLD};
            padding: 18px 22px;
            border-radius: 0 10px 10px 0;
            font-family: 'Cormorant Garamond', serif;
            font-style: italic;
            font-size: 18px;
            color: {PARCHMENT};
        }}
        .niffler-whisper .signed {{
            display: block;
            margin-top: 10px;
            font-family: sans-serif;
            font-style: normal;
            font-size: 11px;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: {GOLD};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


SHOWCASE_URL = "https://github.com/Revanth-Naik/niffler-showcase"


def render_header(tagline: str = "the hoard knows before the bell rings") -> None:
    st.markdown('<div class="niffler-wordmark">NIFFLER</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="niffler-tagline">{tagline}</div>', unsafe_allow_html=True)
    # The Streamlit toolbar's built-in GitHub icon links to the actual
    # deployment source repo, which is private — that link 404s for
    # visitors and can't be redirected (it's wired to the deployment
    # config, not app code). This sidebar link is the real, working one.
    st.sidebar.markdown(
        f'<a href="{SHOWCASE_URL}" target="_blank" style="font-size:12px; color:{GOLD}; '
        f'text-decoration:none;">&#8599; View project on GitHub</a>',
        unsafe_allow_html=True,
    )


def render_prediction_card(ticker: str, price: float, predicted_pct: float, confidence: int, rationale: str = "", source: str = "") -> str:
    up = predicted_pct >= 0
    arrow = "&#8599;" if up else "&#8600;"
    cls = "up" if up else "down"
    sign = "+" if up else ""
    badge_color = GOLD_BRIGHT if source == "ml" else PARCHMENT_DIM
    badge = f'<span style="float:right; font-size:10px; letter-spacing:0.5px; text-transform:uppercase; color:{badge_color}; border:1px solid {HAIRLINE}; border-radius:4px; padding:2px 6px;">{"AI" if source == "ml" else "heuristic"}</span>' if source else ""
    return f"""
    <div class="niffler-card">
        {f'<div style="margin-bottom:6px;">{badge}</div>' if badge else ""}
        <div class="ticker">{ticker} <span style="float:right">{arrow}</span></div>
        <div class="price">${price:,.2f}</div>
        <div class="pct {cls}">{sign}{predicted_pct:.2f}%</div>
        <div style="font-size:11px; color:{PARCHMENT_DIM};">{confidence}% confidence</div>
        {f'<div class="rationale">{rationale}</div>' if rationale else ''}
    </div>
    """


def render_whisper(text: str) -> None:
    st.markdown(
        f'<div class="niffler-whisper">{text}<span class="signed">— Niffler\'s nightly whisper</span></div>',
        unsafe_allow_html=True,
    )


def hoard_gauge(accuracy_pct: float, label: str = "Hoard accuracy") -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=accuracy_pct,
            number={"suffix": "%", "font": {"color": GOLD_BRIGHT, "family": "Cinzel"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": PARCHMENT_DIM},
                "bar": {"color": GOLD_BRIGHT},
                "bgcolor": PANEL,
                "borderwidth": 1,
                "bordercolor": HAIRLINE,
                "steps": [
                    {"range": [0, 50], "color": PANEL_2},
                    {"range": [50, 100], "color": PANEL},
                ],
            },
            title={"text": label, "font": {"color": PARCHMENT_DIM, "size": 13}},
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": PARCHMENT},
        height=260,
        margin=dict(l=20, r=20, t=50, b=10),
    )
    return fig


def themed_line_layout(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": PARCHMENT, "family": "sans-serif"},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        xaxis={"gridcolor": HAIRLINE, "zerolinecolor": HAIRLINE},
        yaxis={"gridcolor": HAIRLINE, "zerolinecolor": HAIRLINE},
    )
    return fig
