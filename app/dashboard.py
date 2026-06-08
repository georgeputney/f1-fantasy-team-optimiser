"""Streamlit dashboard for the F1 fantasy team optimiser."""

import json
from pathlib import Path

import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as st_components

from app.optimiser.optimiser import optimiser
from app.config import REPORTS_DIR, PREDICTIONS_DIR

TEAM_COLORS = {
    "red_bull":      "#3671C6",
    "ferrari":       "#FF2800",
    "mercedes":      "#27F4D2",
    "mclaren":       "#FF8000",
    "aston_martin":  "#229971",
    "alpine":        "#FF87BC",
    "williams":      "#64C4FF",
    "racing_bulls":  "#6692FF",
    "haas":          "#C0C0C0",
    "audi":          "#AA0000",
    "cadillac":      "#F0F0F0",
}

_PLOTLY_THEME = dict(
    paper_bgcolor="#0e0e0d",
    plot_bgcolor="#0e0e0d",
    font=dict(family="DM Sans", color="#f7f6f3"),
    hoverlabel=dict(
        bgcolor="#1c1c1a",
        bordercolor="rgba(247,246,243,0.12)",
        font=dict(family="DM Sans", size=12, color="#f7f6f3"),
    ),
    dragmode=False,
)

_TICK_FONT = dict(family="DM Sans", size=11, color="rgba(247,246,243,0.6)")
_GRID_COLOR = "rgba(247,246,243,0.08)"

_TABLE_CSS = """body { margin:0; background:transparent; font-family:'DM Sans',sans-serif; color:#f7f6f3; }
table { width:100%; border-collapse:collapse; font-family:'DM Sans',sans-serif; font-size:13px; color:#f7f6f3; }
th { color:rgba(247,246,243,0.35); font-size:10px; font-weight:400; letter-spacing:0.1em; text-transform:uppercase; padding:6px 12px 8px 12px; text-align:left; border-bottom:1px solid rgba(247,246,243,0.12); cursor:pointer; user-select:none; white-space:nowrap; }
th:hover { color:rgba(247,246,243,0.65); }
th.sort-asc::after { content:" \u25B2"; font-size:9px; }
th.sort-desc::after { content:" \u25BC"; font-size:9px; }
td { padding:10px 12px; border-bottom:1px solid rgba(247,246,243,0.06); }
tr:last-child td { border-bottom:1px solid rgba(247,246,243,0.12); }
tbody tr:hover td { background:rgba(247,246,243,0.04); }
"""

_SORT_JS_TEMPLATE = r"""(function() {
  var tbl = document.getElementById('__TID__');
  var tbody = tbl.querySelector('tbody');
  var ths = Array.from(tbl.querySelectorAll('thead th'));
  ths.forEach(function(th, i) {
    th.addEventListener('click', function() {
      var asc = th.classList.contains('sort-desc') || !th.classList.contains('sort-asc');
      ths.forEach(function(h) { h.classList.remove('sort-asc','sort-desc'); });
      th.classList.add(asc ? 'sort-asc' : 'sort-desc');
      Array.from(tbody.querySelectorAll('tr'))
        .sort(function(a,b) {
          var at = a.cells[i].innerText.trim(), bt = b.cells[i].innerText.trim();
          var av = parseFloat(at.replace('%','').replace('+','').replace(/[^\d.\-]/g,''));
          var bv = parseFloat(bt.replace('%','').replace('+','').replace(/[^\d.\-]/g,''));
          if (isNaN(av)) av = at.toLowerCase();
          if (isNaN(bv)) bv = bt.toLowerCase();
          return av < bv ? (asc ? -1 : 1) : av > bv ? (asc ? 1 : -1) : 0;
        })
        .forEach(function(r) { tbody.appendChild(r); });
    });
  });
})();"""


def _sort_js(table_id):
    return _SORT_JS_TEMPLATE.replace('__TID__', table_id)


def _sortable_table(table_id, headers_html, rows_html, colgroup_html, height, extra_css=""):
    st_components.html(
        f"<!DOCTYPE html><html><head><style>{_TABLE_CSS}"
        f".scroll-wrap {{ overflow-x:auto; -webkit-overflow-scrolling:touch; }}"
        f"{extra_css}</style></head><body>"
        f'<div class="scroll-wrap"><table id="{table_id}">'
        f"<colgroup>{colgroup_html}</colgroup>"
        f"<thead><tr>{headers_html}</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table></div><script>{_sort_js(table_id)}</script></body></html>",
        height=height, scrolling=False,
    )


st.set_page_config(page_title="F1 Fantasy Optimiser", layout="wide")

# detect viewport width - cookie path is instant (sent with every HTTP request),
# JS round-trip only fires on the very first visit to set the cookie.
_cookie_match = re.search(r'\bvw=(\d+)', st.context.headers.get("cookie", ""))
if _cookie_match:
    st.session_state.viewport_width = int(_cookie_match.group(1))

if "viewport_width" not in st.session_state:
    st_components.html(
        "<script>"
        "var w=window.innerWidth;"
        "document.cookie='vw='+w+';path=/;max-age=31536000';"
        "window.parent.location.reload();"
        "</script>",
        height=0,
    )

_vw_known = "viewport_width" in st.session_state
_is_mobile: bool = st.session_state.get("viewport_width", 1920) < 768

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap');

/* Hide Streamlit chrome */
[data-testid="stHeadingAnchorLink"], [data-testid="stToolbar"], [data-testid="stElementToolbar"], footer, #MainMenu { display: none !important; }

/* Hide "Select all" and collapse menu when only max-selections message remains */
[data-baseweb="menu"] ul li:first-child { display: none !important; }
[data-baseweb="menu"] [aria-label="Select all"] { display: none !important; }
[data-baseweb="menu"] li[aria-disabled="true"] { display: none !important; }
[data-baseweb="menu"] ul:not(:has(li:not([aria-disabled="true"]))) { display: none !important; }

/* Body font */
html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }

/* h1 gets the portfolio serif */
h1 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    letter-spacing: -0.02em !important;
}

/* h2/h3 also get serif - covers subheader() and ### Recommended team */
h2, h3 {
    font-family: 'Cormorant Garamond', serif !important;
    font-weight: 300 !important;
    letter-spacing: -0.01em !important;
}

/* Caption as a section label */
[data-testid="stCaptionContainer"] p {
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase;
    color: rgba(247,246,243,0.35) !important;
}

/* Tighten default page padding */
.main .block-container { padding-top: 0.5rem; padding-left: 1rem !important; padding-right: 1rem !important; }

/* Tabs - smaller, right-aligned */
.stTabs [data-baseweb="tab"] {
    font-size: 10px !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase;
    font-weight: 500;
    color: rgba(247,246,243,0.4);
    border: none;
    padding: 0 0 3px 0;
}
.stTabs [data-baseweb="tab-list"] { gap: 24px; border-bottom: 1px solid rgba(247,246,243,0.12); justify-content: flex-end; }
@media (max-width: 768px) {
    .stTabs [data-baseweb="tab-list"] { display:flex !important; width:100% !important; justify-content:space-between !important; gap:0 !important; border-bottom:1px solid rgba(247,246,243,0.12); }
    .stTabs [data-baseweb="tab"] { flex:1 !important; min-width:0 !important; text-align:center !important; justify-content:center !important; align-items:center !important; display:flex !important; padding:10px 2px !important; font-size:9px !important; letter-spacing:0.06em !important; }
    input, select, textarea { font-size: 16px !important; }
}
.stTabs [aria-selected="true"] { color: #c8401a !important; border-bottom: 1px solid #c8401a !important; }

/* Metrics use the serif for editorial feel */
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 300;
    color: #f7f6f3;
}
[data-testid="stMetricLabel"] {
    font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: rgba(247,246,243,0.4);
}

/* Table headers as section labels */
[data-testid="stDataFrame"] th {
    font-size: 10px !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: rgba(247,246,243,0.4) !important; font-weight: 400 !important;
}

/* Plotly - pointer cursor instead of crosshair */
.js-plotly-plot .plotly .cursor-crosshair { cursor: pointer !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>F1 Fantasy<br/>Team Optimiser</h1>", unsafe_allow_html=True)



# formats driver/constructor IDs as display names e.g. max_verstappen -> Max Verstappen
def format_name(id_str):
    return id_str.replace("_", " ").title()

def last_name(id_str):
    return id_str.split("_")[-1].title()


tab1, tab2, tab3 = st.tabs(
    ["Team Picker", "Performance", "Breakdown"] if _is_mobile else
    ["Team Picker", "Model Performance", "Driver Breakdown"]
)


# team picker tab
with tab1:
    _available = sorted(PREDICTIONS_DIR.glob("predictions_????_??.json"))
    if not _available:
        st.info("No predictions available yet. Run `generate-reports` to create them.")
        st.stop()
    pred_path = _available[-1]  # highest season + round

    data = json.loads(pred_path.read_text())
    driver_team = {d["driver_id"]: d["constructor_id"] for d in data["drivers"] if "constructor_id" in d}

    st.markdown(
        f'<p style="font-family:\'Cormorant Garamond\',serif;font-size:1.8rem;font-weight:300;letter-spacing:-0.01em;color:#f7f6f3;margin:0 0 2px 0">'
        f'R{data["round"]}: {data["circuit"]}</p>'
        f'<p style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:rgba(247,246,243,0.35);margin:0 0 1rem 0">'
        f'Last updated {data["generated_at"][:10]}</p>',
        unsafe_allow_html=True,
    )

    driver_df = pd.DataFrame(data["drivers"]).rename(columns={"expected_points": "expected_fantasy_points"})
    constructor_df = pd.DataFrame(data["constructors"]).rename(columns={"expected_points": "expected_fantasy_points"})
    prices_df = pd.DataFrame(
        [{"asset_id": d["driver_id"], "price": d["price"]} for d in data["drivers"]] +
        [{"asset_id": c["constructor_id"], "price": c["price"]} for c in data["constructors"]]
    )
    prices_index = prices_df.set_index("asset_id")["price"]

    col_left, col_spacer, col_right = st.columns([1, 0.1, 2])

    with col_left:
        budget_error_slot = st.empty()
        budget = st.number_input("Budget (£M)", min_value=0.0, max_value=200.0, value=100.0, step=0.1, format="%.1f")
        def _err(msg, slot):
            slot.markdown(f'<p style="font-family:\'DM Sans\',sans-serif;font-size:13px;color:#c8401a;margin:0 0 8px 0">{msg}</p>', unsafe_allow_html=True)

        has_current_team = st.checkbox("Enter current team")
        driver_error_slot = st.empty()
        current_drivers = []
        current_constructors = []
        free_transfers = 2
        if has_current_team:
            current_drivers = st.pills(
                "Current drivers",
                options=[d["driver_id"] for d in sorted(data["drivers"], key=lambda x: x["price"], reverse=True)],
                format_func=format_name,
                selection_mode="multi",
                default=[],
            )
            if len(current_drivers) > 5:
                _err("Select at most 5 drivers.", driver_error_slot)
                st.stop()
            constructor_error_slot = st.empty()
            current_constructors = st.pills(
                "Current constructors",
                options=[c["constructor_id"] for c in sorted(data["constructors"], key=lambda x: x["price"], reverse=True)],
                format_func=format_name,
                selection_mode="multi",
                default=[],
            )
            if len(current_constructors) > 2:
                _err("Select at most 2 constructors.", constructor_error_slot)
                st.stop()
            team_cost = sum(float(prices_index.get(i, 0)) for i in current_drivers + current_constructors)
            if team_cost > 0:
                remaining = budget - team_cost
                if remaining < 0:
                    _err(f"Team costs £{team_cost:.1f}M, £{abs(remaining):.1f}M over budget.", budget_error_slot)
                    st.stop()
                st.progress(min(team_cost / budget, 1.0), text=f"£{team_cost:.1f}M spent - £{remaining:.1f}M remaining")

            free_transfers = st.radio("Free transfers", [2, 3], horizontal=True)

        state = None
        if current_drivers or current_constructors:
            state = {
                "drivers": current_drivers,
                "constructors": current_constructors,
                "prices": {i: float(prices_index.get(i, 0)) for i in current_drivers + current_constructors},
                "budget_remaining": budget - team_cost,
                "free_transfers_carried": free_transfers - 2,
            }

        team = optimiser(driver_df, constructor_df, prices_df, budget, state)
        selected_ids = set(team["drivers"] + team["constructors"])

        st.markdown("### Recommended team")

        total_points = 0.0
        total_cost = 0.0

        driver_rows = []
        for d in team["drivers"]:
            points = float(driver_df.set_index("driver_id")["expected_fantasy_points"][d])
            price = float(prices_index[d])
            doubled = d == team["doubled_driver"]
            display_points = points * 2 if doubled else points
            team_color = TEAM_COLORS.get(driver_team.get(d, ""), "#888")
            driver_rows.append({
                "id": d,
                "Driver": format_name(d),
                "suffix": " x2" if doubled else "",
                "points": display_points,
                "Price": f"£{price:.1f}M",
                "color": team_color,
                "doubled": doubled,
            })
            total_points += display_points
            total_cost += price

        constructor_rows = []
        for c in team["constructors"]:
            points = float(constructor_df.set_index("constructor_id")["expected_fantasy_points"][c])
            price = float(prices_index[c])
            team_color = TEAM_COLORS.get(c, "#888")
            constructor_rows.append({
                "Constructor": format_name(c),
                "points": points,
                "Price": f"£{price:.1f}M",
                "color": team_color,
            })
            total_points += points
            total_cost += price

        def _points_html(points):
            if points < 0:
                color = "#e05252"
            elif points >= 50:
                color = "#f7f6f3"
            else:
                color = "rgba(247,246,243,0.75)"
            return f'<span style="color:{color};font-variant-numeric:tabular-nums">{points:.1f}</span>'

        def _dot(color):
            return f'<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:{color};margin-right:9px;vertical-align:middle"></span>'

        def _tr_driver(r):
            suffix = f' <span style="color:rgba(247,246,243,0.45);font-size:11px">x2</span>' if r["suffix"] else ""
            return (
                f'<tr>'
                f'<td>{_dot(r["color"])}{r["Driver"]}{suffix}</td>'
                f'<td>{_points_html(r["points"])}</td>'
                f'<td>{r["Price"]}</td>'
                f'</tr>'
            )

        def _tr_constructor(r):
            return (
                f'<tr>'
                f'<td>{_dot(r["color"])}{r["Constructor"]}</td>'
                f'<td>{_points_html(r["points"])}</td>'
                f'<td>{r["Price"]}</td>'
                f'</tr>'
            )

        driver_rows_html = "".join(_tr_driver(r) for r in driver_rows)
        constructor_rows_html = "".join(_tr_constructor(r) for r in constructor_rows)
        extra = team["transfer_penalty"]
        penalty_note = f' <span style="font-size:1rem;color:rgba(247,246,243,0.45);font-family:\'DM Sans\',sans-serif;font-weight:300">(-{extra * 10}pt, {extra} extra transfer{"s" if extra != 1 else ""})</span>' if state and extra > 0 else ""

        table_height = 32 + len(driver_rows) * 41 + 16 + 32 + len(constructor_rows) * 41 + 216

        _col_group = '<col class="name"><col class="stat"><col class="stat">'
        table_html = (
            "<!DOCTYPE html><html><head><style>"
            "@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300&family=DM+Sans&display=swap');"
            + _TABLE_CSS
            + "col.name { width:55%; } col.stat { width:22%; }"
            " .stat-label { font-size:10px; letter-spacing:0.1em; text-transform:uppercase; color:rgba(247,246,243,0.4); margin:1.8rem 0 4px 0; }"
            " .stat-value { font-family:'Cormorant Garamond',serif; font-size:2.4rem; font-weight:300; margin:0 0 0 0; }"
            "</style></head><body>"
            f'<table id="tbl-drivers"><colgroup>{_col_group}</colgroup>'
            f"<thead><tr><th>Driver</th><th>Pts</th><th>Price</th></tr></thead>"
            f"<tbody>{driver_rows_html}</tbody></table>"
            '<div style="height:1rem"></div>'
            f'<table id="tbl-constructors"><colgroup>{_col_group}</colgroup>'
            f"<thead><tr><th>Constructor</th><th>Pts</th><th>Price</th></tr></thead>"
            f"<tbody>{constructor_rows_html}</tbody></table>"
            f'<p class="stat-label" style="margin-top:3rem">Projected points</p>'
            f'<p class="stat-value">{total_points:.1f}{penalty_note}</p>'
            f'<p class="stat-label">Total cost</p>'
            f'<p class="stat-value">\u00a3{total_cost:.1f}M <span style="font-size:1rem;color:rgba(247,246,243,0.45);'
            f"font-family:'DM Sans',sans-serif;font-weight:300\">/ \u00a3{budget:.1f}M</span></p>"
            f"<script>{_sort_js('tbl-drivers')}{_sort_js('tbl-constructors')}</script>"
            "</body></html>"
        )
        st_components.html(table_html, height=table_height, scrolling=False)

    with col_right:
        if not _vw_known:
            st.markdown(
                '<p style="color:rgba(247,246,243,0.2);font-size:11px;letter-spacing:0.1em;'
                'text-transform:uppercase;margin-top:6rem;text-align:center">Loading…</p>',
                unsafe_allow_html=True,
            )
        else:
            # per-chart x ranges with 0 aligned at the same relative position in both charts
            d_left = abs(min(driver_df["expected_fantasy_points"].min(), 0)) * 1.1
            d_right = driver_df["expected_fantasy_points"].max() * 1.05
            zero_fraction = d_left / (d_left + d_right) if (d_left + d_right) > 0 else 0

            c_right = constructor_df["expected_fantasy_points"].max() * 1.05
            c_left = (zero_fraction * c_right / (1 - zero_fraction)) if zero_fraction > 0 else 0

            driver_x_range = [-d_left, d_right]
            constructor_x_range = [-c_left, c_right]

            _name_fn = last_name if _is_mobile else format_name
            _chart_l = 80 if _is_mobile else 130

            # drivers bar chart - sorted by expected points, selected team highlighted in red with bold label
            driver_chart = driver_df[["driver_id", "expected_fantasy_points", "price"]].assign(
                name=lambda df: df["driver_id"].apply(lambda x: f"<b>{_name_fn(x)}</b>" if x in selected_ids else _name_fn(x)),
                color=lambda df: df["driver_id"].apply(lambda x: TEAM_COLORS.get(driver_team.get(x, ""), "#f7f6f3") if x in selected_ids else "#3a3a38"),
            ).sort_values("expected_fantasy_points")

            fig_drivers = px.bar(
                driver_chart,
                x="expected_fantasy_points",
                y="name",
                orientation="h",
                hover_data={"price": False, "color": False, "name": False, "expected_fantasy_points": False},
                labels={"expected_fantasy_points": "Expected Points", "name": "", "price": "Price (£M)"},
                custom_data=["price"],
            )
            fig_drivers.update_traces(
                marker_color=driver_chart["color"].tolist(),
                hovertemplate="<b>%{y}</b><br>Expected points: %{x:.1f}<br>Price: £%{customdata[0]:.1f}M<extra></extra>",
            )
            fig_drivers.update_layout(
                **_PLOTLY_THEME,
                showlegend=False, height=500, margin=dict(l=_chart_l, r=0, t=36, b=0),
                title=dict(text="Drivers", font=dict(size=11, color="rgba(247,246,243,0.4)", family="DM Sans"), x=0),
                xaxis=dict(gridcolor=_GRID_COLOR, range=driver_x_range, fixedrange=True,
                           title_font=_TICK_FONT, tickfont=_TICK_FONT),
                yaxis=dict(automargin=False, fixedrange=True, tickfont=_TICK_FONT),
                hovermode="closest",
                clickmode="none",
            )
            st.plotly_chart(fig_drivers, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

            # constructors bar chart - selected team highlighted in red with bold label
            constructor_chart = constructor_df[["constructor_id", "expected_fantasy_points", "price"]].assign(
                name=lambda df: df["constructor_id"].apply(lambda x: f"<b>{format_name(x)}</b>" if x in selected_ids else format_name(x)),
                color=lambda df: df["constructor_id"].apply(lambda x: TEAM_COLORS.get(x, "#f7f6f3") if x in selected_ids else "#3a3a38"),
            ).sort_values("expected_fantasy_points")

            fig_constructors = px.bar(
                constructor_chart,
                x="expected_fantasy_points",
                y="name",
                orientation="h",
                hover_data={"price": False, "color": False, "name": False, "expected_fantasy_points": False},
                labels={"expected_fantasy_points": "Expected Points", "name": "", "price": "Price (£M)"},
                custom_data=["price"],
            )
            fig_constructors.update_traces(
                marker_color=constructor_chart["color"].tolist(),
                hovertemplate="<b>%{y}</b><br>Expected points: %{x:.1f}<br>Price: £%{customdata[0]:.1f}M<extra></extra>",
            )
            fig_constructors.update_layout(
                **_PLOTLY_THEME,
                showlegend=False, height=300, margin=dict(l=_chart_l, r=0, t=36, b=0),
                title=dict(text="Constructors", font=dict(size=11, color="rgba(247,246,243,0.4)", family="DM Sans"), x=0),
                xaxis=dict(gridcolor=_GRID_COLOR, range=constructor_x_range, fixedrange=True,
                           title_font=_TICK_FONT, tickfont=_TICK_FONT),
                yaxis=dict(automargin=False, fixedrange=True, tickfont=_TICK_FONT),
                hovermode="closest",
                clickmode="none",
            )
            st.plotly_chart(fig_constructors, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})


# model performance tab
with tab2:
    backtest_files = sorted(REPORTS_DIR.glob("backtest_*.json"))

    if not backtest_files:
        st.info("No backtest data available. Run `backtest` to generate it.")
        st.stop()

    dfs = []
    for f in backtest_files:
        season_num = int(f.stem.split("_")[1])
        df_bt = pd.DataFrame(json.loads(f.read_text()))
        df_bt["season"] = season_num
        dfs.append(df_bt)

    all_data = pd.concat(dfs).reset_index(drop=True)
    all_data = all_data[all_data["season"] >= 2026]

    summary = (
        all_data.groupby("season")[["model", "oracle"]]
        .sum()
        .assign(pct_of_oracle=lambda df: (df["model"] / df["oracle"] * 100).round(1))
        .rename(columns={"model": "Model (our prediction)", "oracle": "Oracle (best possible)", "pct_of_oracle": "% of Oracle"})
    )
    summary.index.name = "Season"

    def _pct_bar(pct):
        return (
            f'<div style="display:flex;align-items:center;gap:10px">'
            f'<div style="flex:1;height:3px;background:rgba(247,246,243,0.08);border-radius:2px">'
            f'<div style="width:{pct:.1f}%;height:100%;background:#c8401a;border-radius:2px"></div>'
            f'</div>'
            f'<span style="min-width:3.5em;text-align:right">{pct:.1f}%</span>'
            f'</div>'
        )

    st.subheader("Season summary")
    summary_rows = "".join(
        f'<tr><td>{int(season)}</td><td>{int(row["Model (our prediction)"])}</td><td>{int(row["Oracle (best possible)"])}</td><td>{_pct_bar(row["% of Oracle"])}</td></tr>'
        for season, row in summary.iterrows()
    )
    if _is_mobile:
        _sum_headers = ["Season", "Model", "Oracle", "% Oracle"]
        _sum_colgroup = '<col style="width:16%"><col style="width:18%"><col style="width:18%"><col style="width:48%">'
    else:
        _sum_headers = ["Season", "Model (our prediction)", "Oracle (best possible)", "% of Oracle"]
        _sum_colgroup = '<col style="width:15%"><col style="width:20%"><col style="width:20%"><col style="width:45%">'
    _sortable_table("stbl", "".join(f'<th>{h}</th>' for h in _sum_headers),
                    summary_rows, _sum_colgroup, 32 + len(summary) * 41 + 2)

    st.subheader("Cumulative points by strategy")
    season_options = sorted(s for s in all_data["season"].unique() if s >= 2026)
    selected_season = st.selectbox("Season", season_options, index=len(season_options) - 1, key="backtest_season")

    season_data = all_data[all_data["season"] == selected_season].copy()
    season_data["Model (our prediction)"] = season_data["model"].cumsum()
    season_data["Oracle (best possible)"] = season_data["oracle"].cumsum()
    has_location = "location" in season_data.columns

    id_vars = ["round"] + (["location"] if has_location else [])
    cumulative_melted = season_data.melt(id_vars=id_vars, value_vars=["Model (our prediction)", "Oracle (best possible)"],
                                         var_name="Strategy", value_name="Cumulative Points")
    round_melted = season_data.melt(id_vars=["round"], value_vars=["model", "oracle"],
                                    var_name="_s", value_name="round_points")
    round_melted["Strategy"] = round_melted["_s"].map({"model": "Model (our prediction)", "oracle": "Oracle (best possible)"})
    round_melted["short_name"] = round_melted["_s"].str.capitalize()
    melted = cumulative_melted.merge(round_melted[["round", "Strategy", "round_points", "short_name"]], on=["round", "Strategy"])

    custom = (["location", "round_points", "short_name"] if has_location else ["round_points", "short_name"])
    fig2 = px.line(
        melted,
        x="round",
        y="Cumulative Points",
        color="Strategy",
        color_discrete_map={"Model (our prediction)": "#c8401a", "Oracle (best possible)": "rgba(247,246,243,0.4)"},
        labels={"round": "Round"},
        markers=True,
        custom_data=custom,
    )
    fig2.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>%{customdata[2]}: %{y:,.0f} points (+%{customdata[1]:.0f} this round)<extra></extra>"
            if has_location else
            "%{customdata[1]}: %{y:,.0f} points (+%{customdata[0]:.0f} this round)<extra></extra>"
        )
    )
    fig2.update_layout(
        **_PLOTLY_THEME,
        xaxis=dict(gridcolor=_GRID_COLOR, tickmode="linear", dtick=1,
                   title_font=_TICK_FONT, tickfont=_TICK_FONT),
        yaxis=dict(gridcolor=_GRID_COLOR, title_font=_TICK_FONT, tickfont=_TICK_FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    title_text="", font=_TICK_FONT),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    # per-round % of oracle achieved
    season_data["pct_of_oracle"] = (season_data["model"] / season_data["oracle"] * 100).round(1)
    max_round = all_data["round"].max()
    fig3 = px.bar(
        season_data,
        x="round",
        y="pct_of_oracle",
        custom_data=["location", "model", "oracle"] if has_location else ["model", "oracle"],
        labels={"round": "Round", "pct_of_oracle": "% of Oracle"},
        title="% of oracle achieved per round",
    )
    hover = (
        "<b>%{customdata[0]}</b><br>% of Oracle: %{y:.1f}%<br>Model points: %{customdata[1]}<br>Oracle points: %{customdata[2]}<extra></extra>"
        if has_location else
        "% of Oracle: %{y:.1f}%<br>Model points: %{customdata[0]}<br>Oracle points: %{customdata[1]}<extra></extra>"
    )
    fig3.update_traces(marker_color="#c8401a", hovertemplate=hover)
    fig3.add_hline(y=season_data["pct_of_oracle"].mean(), line_dash="dash", line_color="rgba(247,246,243,0.3)",
                   annotation_text=f"avg {season_data['pct_of_oracle'].mean():.1f}%", annotation_position="top right")
    fig3.add_hline(y=100, line_dash="solid", line_color="rgba(247,246,243,0.08)", line_width=1)
    fig3.update_yaxes(range=[0, 105])
    fig3.update_xaxes(range=[0.5, max_round + 0.5], tickmode="linear", dtick=1)
    fig3.update_layout(
        **_PLOTLY_THEME,
        margin=dict(t=40),
        xaxis=dict(gridcolor=_GRID_COLOR, fixedrange=True,
                   title_font=_TICK_FONT, tickfont=_TICK_FONT),
        yaxis=dict(gridcolor=_GRID_COLOR, fixedrange=True,
                   title_font=_TICK_FONT, tickfont=_TICK_FONT),
        hovermode="closest",
        clickmode="none",
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})


# driver breakdown tab
with tab3:
    # collect all versioned prediction files e.g. predictions_2024_01.json
    versioned = sorted(PREDICTIONS_DIR.glob("predictions_????_??.json"))

    if versioned:
        # build {season: [(round, circuit), ...]} index — read circuit from each file header
        _pred_index: dict[int, list[tuple[int, str]]] = {}
        for _p in versioned:
            _parts = _p.stem.split("_")  # ['predictions', 'YYYY', 'RR']
            _s, _r = int(_parts[1]), int(_parts[2])
            _meta = json.loads(_p.read_text())
            _circuit = _meta.get("circuit", f"Round {_r}")
            _pred_index.setdefault(_s, []).append((_r, _circuit))
        for _s in _pred_index:
            _pred_index[_s].sort(key=lambda x: x[0])

        _seasons = sorted(_pred_index.keys(), reverse=True)
        _col1, _col2 = st.columns(2)
        with _col1:
            _sel_season = st.selectbox("Season", _seasons, index=0, key="breakdown_season")
        with _col2:
            _round_options = _pred_index[_sel_season]
            _round_labels = [f"R{r}: {c}" for r, c in _round_options]
            _round_idx = st.selectbox("Round", range(len(_round_options)), index=len(_round_options) - 1, format_func=lambda i: _round_labels[i])
            _sel_round = _round_options[_round_idx][0]

        pred_path = PREDICTIONS_DIR / f"predictions_{_sel_season}_{_sel_round:02d}.json"
    else:
        # fall back to latest if no versioned files exist yet
        pred_path = PREDICTIONS_DIR / "predictions_latest.json"

    if not pred_path.exists():
        st.info("No predictions available yet. Run `generate-reports` to create them.")
        st.stop()

    data = json.loads(pred_path.read_text())

    # skip if JSON predates breakdown support
    if not data["drivers"] or "points_breakdown" not in data["drivers"][0]:
        st.info("No breakdown data available. Re-run `generate-reports` to include it.")
        st.stop()

    _BREAKDOWN_KEYS = ["finish", "quali", "positions_gained", "overtakes", "prob_fl", "prob_dotd", "sprint"]

    breakdown_rows = []
    for d in data["drivers"]:
        bd = d.get("points_breakdown", {})
        breakdown_rows.append({
            "driver_id": d["driver_id"],
            "constructor_id": d["constructor_id"],
            "name": format_name(d["driver_id"]),
            "price": d.get("price", 0.0),
            "total": d["expected_points"],
            "quali_pos": d["predicted_quali_position"],
            "finish_pos": d["predicted_finish_position"],
            **{k: bd.get(k, 0.0) for k in _BREAKDOWN_KEYS},
        })

    breakdown_df = pd.DataFrame(breakdown_rows).sort_values("total", ascending=False)

    # parallel coordinates chart (desktop only)
    # axes: Grid pos, Finish pos, Pos gain, FL prob, Expected points
    # positional axes are inverted so "better" (lower number) is always at the top
    _PC_DIMS = [
        ("quali_pos",        "Qualifying",    True,  lambda v: f"P{int(v)}"),
        ("finish_pos",       "Finish",        True,  lambda v: f"P{int(v)}"),
        ("positions_gained", "Pos \u00b1",    False, lambda v: f"{int(v):+d}"),
        ("overtakes",        "Overtakes",     False, lambda v: f"{int(v)}"),
        ("prob_fl",          "FL (prob)",     False, lambda v: f"{v:.1%}"),
        ("prob_dotd",        "DOTD (prob)",   False, lambda v: f"{v:.1%}"),
        ("total",            "Pts",           False, lambda v: f"{v:.1f}"),
    ]

    def _norm_fn(mn, mx, invert):
        if mx == mn:
            return lambda _: 0.5
        if invert:
            return lambda x, _mn=mn, _mx=mx: 1.0 - (float(x) - _mn) / (_mx - _mn)
        return lambda x, _mn=mn, _mx=mx: (float(x) - _mn) / (_mx - _mn)

    col_stats = {}
    for col, _, invert, _ in _PC_DIMS:
        vals = breakdown_df[col].astype(float)
        mn, mx = float(vals.min()), float(vals.max())
        col_stats[col] = (mn, mx, _norm_fn(mn, mx, invert), invert)

    top_ids = set(breakdown_df.nlargest(6, "total")["driver_id"])
    n_axes = len(_PC_DIMS)

    # cheaper driver per team gets a dashed line (proxy for secondary driver)
    _team_drivers: dict = {}
    for _, row in breakdown_df.iterrows():
        cid = row["constructor_id"]
        _team_drivers.setdefault(cid, []).append((row["price"], row["driver_id"]))
    dashed_drivers = {
        min(drivers, key=lambda x: x[0])[1]
        for drivers in _team_drivers.values()
        if len(drivers) == 2
    }

    # per-segment smoothstep (cubic Hermite, horizontal tangents at every axis)
    # this is the classic parallel-coordinates curve: lines ease in/out at each axis
    # with zero slope, so they never overshoot - unlike Catmull-Rom which can loop
    # wildly when adjacent axes have the same value (e.g. all-zero Overtakes).
    _points_PER_SEG = 20
    _INTERP_points = (n_axes - 1) * _points_PER_SEG + 1
    def _smoothstep_spline(y_ctrl):
        xs, ys = [], []
        for i in range(n_axes - 1):
            t = np.linspace(0, 1, _points_PER_SEG, endpoint=False)
            s = 3*t**2 - 2*t**3          # smoothstep: 0→1, zero derivative at both ends
            xs.extend((i + t).tolist())
            ys.extend(np.clip(y_ctrl[i] + (y_ctrl[i + 1] - y_ctrl[i]) * s, 0.0, 1.0).tolist())
        xs.append(float(n_axes - 1))
        ys.append(float(np.clip(y_ctrl[-1], 0.0, 1.0)))
        return xs, ys

    fig_pc = go.Figure()

    # non-top drivers drawn first so top drivers render on top
    draw_order = breakdown_df.copy()
    draw_order["_top"] = draw_order["driver_id"].isin(top_ids).astype(int)
    draw_order = draw_order.sort_values("_top")

    # team color for every trace in draw order (used by hover JS)
    trace_team_colors = []

    for _, row in draw_order.iterrows():
        is_top = row["driver_id"] in top_ids
        color = TEAM_COLORS.get(row["constructor_id"], "#888888") if is_top else "rgba(247,246,243,0.1)"
        width = 2.5 if is_top else 1
        dash = "dash" if row["driver_id"] in dashed_drivers else "solid"
        trace_team_colors.append(TEAM_COLORS.get(row["constructor_id"], "#888888"))

        y_vals = [col_stats[col][2](row[col]) for col, _, _, _ in _PC_DIMS]
        x_interp, y_interp = _smoothstep_spline(y_vals)

        pg = int(row["positions_gained"])
        pg_color = "#4ade80" if pg > 0 else ("#f87171" if pg < 0 else "rgba(247,246,243,0.35)")
        _dim = "color:rgba(247,246,243,0.4)"
        hover_html = (
            f"<b style='font-size:13px'>{row['name']}</b>"
            f"<br><span style='{_dim}'>Quali</span>&nbsp;P{int(row['quali_pos'])}"
            f"&nbsp;<span style='{_dim}'>→</span>&nbsp;"
            f"<span style='{_dim}'>Finish</span>&nbsp;P{int(row['finish_pos'])}"
            f"&nbsp;<span style='color:{pg_color}'>({pg:+d})</span>"
            f"<br><span style='{_dim}'>OT</span>&nbsp;{row['overtakes']:.1f}"
            f"&nbsp;&nbsp;<span style='{_dim}'>FL</span>&nbsp;{row['prob_fl']:.1%}"
            f"&nbsp;&nbsp;<span style='{_dim}'>DOTD</span>&nbsp;{row['prob_dotd']:.1%}"
            f"<br><b style='font-size:13px'>{row['total']:.1f} points</b>"
        )
        custom = [[hover_html]] * _INTERP_points

        fig_pc.add_trace(go.Scatter(
            x=x_interp,
            y=y_interp,
            mode="lines+markers",
            marker=dict(size=12, color="rgba(0,0,0,0)", line=dict(width=0)),
            line=dict(color=color, width=width, shape="linear", dash=dash),
            name=row["name"],
            customdata=custom,
            hovertemplate="%{customdata[0]}<extra></extra>",
            showlegend=False,
        ))

    # axis lines, labels, and tick values
    _INT_COLS = {"quali_pos", "finish_pos", "positions_gained", "overtakes"}

    for i, (col, label, _, fmt) in enumerate(_PC_DIMS):
        mn, mx, _, inv = col_stats[col]
        top_val = mn if inv else mx   # value at y=1 (best)
        bot_val = mx if inv else mn   # value at y=0 (worst)

        if col in _INT_COLS:
            # inverted axes: top is minimum (best position), so floor; bottom is maximum, so ceil
            # normal axes: top is maximum (best value), so ceil; bottom is minimum, so floor
            top_text = fmt(int(np.floor(top_val)) if inv else int(np.ceil(top_val)))
            bot_text = fmt(int(np.ceil(bot_val)) if inv else int(np.floor(bot_val)))
        else:
            top_text = fmt(top_val)
            bot_text = fmt(bot_val)

        fig_pc.add_shape(
            type="line", x0=i, y0=-0.02, x1=i, y1=1.02,
            line=dict(color="rgba(247,246,243,0.2)", width=1),
        )
        _lbl_size = 10
        _tick_size = 9
        fig_pc.add_annotation(
            x=i, y=1.18, text=label.upper(), showarrow=False, xanchor="center",
            font=dict(family="DM Sans", size=_lbl_size, color="rgba(247,246,243,0.4)"),
        )
        fig_pc.add_annotation(
            x=i, y=1.07, text=top_text, showarrow=False, xanchor="center",
            font=dict(family="DM Sans", size=_tick_size, color="rgba(247,246,243,0.28)"),
        )
        fig_pc.add_annotation(
            x=i, y=-0.09, text=bot_text, showarrow=False, xanchor="center",
            font=dict(family="DM Sans", size=_tick_size, color="rgba(247,246,243,0.28)"),
        )

    _pc_height = 700
    fig_pc.update_layout(
        **_PLOTLY_THEME,
        height=_pc_height,
        margin=dict(l=10, r=20, t=4, b=30),
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                   range=[-0.3, n_axes - 0.7], fixedrange=True),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False,
                   range=[-0.22, 1.28], fixedrange=True),
        hovermode="closest",
    )

    if not _is_mobile:
        st.subheader("Performance profiles")
        fig_json = fig_pc.to_json()
        trace_colors_js = json.dumps(trace_team_colors)
        pc_html = f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{ font-family:'DM Sans',system-ui,sans-serif; }}
body {{ margin:0; padding:0; background:#0e0e0d; }}
#pc {{ width:100%; height:{_pc_height}px; opacity:0; transition:opacity 0.15s; }}
</style>
</head>
<body>
<div id="pc"></div>
<script>
var fig = {fig_json};
var teamColors = {trace_colors_js};
Plotly.newPlot('pc', fig.data, fig.layout, {{displayModeBar:false, scrollZoom:false, responsive:true}});
var div = document.getElementById('pc');
window.addEventListener('load', function() {{
    window.dispatchEvent(new Event('resize'));
    setTimeout(function() {{ div.style.opacity = '1'; }}, 120);
}});
var origColors = fig.data.map(function(t) {{ return t.line ? t.line.color : 'rgba(247,246,243,0.1)'; }});
var origWidths = fig.data.map(function(t) {{ return (t.line && t.line.width) ? t.line.width : 1; }});
div.on('plotly_hover', function(evt) {{
    var idx = evt.points[0].curveNumber;
    var nc = origColors.map(function(c, i) {{ return i === idx ? teamColors[i] : 'rgba(247,246,243,0.04)'; }});
    var nw = origWidths.map(function(w, i) {{ return i === idx ? 2.5 : 0.5; }});
    Plotly.restyle(div, {{'line.color': nc, 'line.width': nw}});
}});
div.on('plotly_unhover', function() {{
    Plotly.restyle(div, {{'line.color': origColors, 'line.width': origWidths}});
}});
</script>
</body>
</html>"""

        top_rows = breakdown_df[breakdown_df["driver_id"].isin(top_ids)].sort_values("total", ascending=False)
        legend_html_items = []
        for _, r in top_rows.iterrows():
            tc = TEAM_COLORS.get(r["constructor_id"], "#888888")
            dash_attr = 'stroke-dasharray="4 3"' if r["driver_id"] in dashed_drivers else ""
            swatch = (
                f'<svg width="22" height="10" style="vertical-align:middle;margin-right:5px">'
                f'<line x1="0" y1="5" x2="22" y2="5" stroke="{tc}" stroke-width="2" {dash_attr}/>'
                f'</svg>'
            )
            legend_html_items.append(
                f'<span style="display:inline-flex;align-items:center">{swatch}{r["name"]}</span>'
            )
        legend_div = (
            '<div style="display:flex;justify-content:center;align-items:center;gap:22px;'
            'padding:4px 0 0 0;font-family:\'DM Sans\',system-ui,sans-serif;'
            'font-size:12px;color:rgba(247,246,243,0.55)">'
            + "".join(legend_html_items)
            + "</div>"
        )
        pc_html = pc_html.replace('<div id="pc">', legend_div + '\n<div id="pc">')
        st_components.html(pc_html, height=_pc_height + 38, scrolling=False)

    # detail table
    st.subheader("Predictions breakdown")
    has_sprint_col = any(d.get("points_breakdown", {}).get("sprint", 0) != 0 for d in data["drivers"])
    table_rows = ""
    for _, row in breakdown_df.sort_values("total", ascending=False).iterrows():
        team_color = TEAM_COLORS.get(row["constructor_id"], "#888")
        sprint_cell = f'<td>{row["sprint"]:.1f}</td>' if has_sprint_col else ""
        pos_gained = row["positions_gained"]
        pos_color = "#27ae60" if pos_gained > 0 else ("#e05252" if pos_gained < 0 else "rgba(247,246,243,0.4)")
        driver_label = last_name(row["driver_id"]) if _is_mobile else row["name"]
        table_rows += (
            f'<tr>'
            f'<td>{_dot(team_color)}{driver_label}</td>'
            f'<td style="color:rgba(247,246,243,0.6)">{row["quali_pos"]}</td>'
            f'<td style="color:rgba(247,246,243,0.6)">{row["finish_pos"]}</td>'
            f'<td style="color:{pos_color}">{pos_gained:+.0f}</td>'
            f'<td style="color:rgba(247,246,243,0.55)">{row["overtakes"]:.1f}</td>'
            f'<td style="color:rgba(247,246,243,0.55)">{row["prob_fl"]:.1%}</td>'
            f'<td style="color:rgba(247,246,243,0.55)">{row["prob_dotd"]:.1%}</td>'
            f'{sprint_cell}'
            f'<td style="font-weight:500">{row["total"]:.1f}</td>'
            f'</tr>'
        )

    sprint_header = "<th>Sprint</th>" if has_sprint_col else ""
    _detail_css = "" if _is_mobile else "td, th { white-space:nowrap; } "
    if _is_mobile:
        _detail_css += (
            "table { border-collapse:separate !important; border-spacing:0; min-width:0; } "
            "th, td { white-space:nowrap; min-width:55px; overflow:hidden; } "
            "th:first-child, td:first-child { min-width:86px; max-width:86px; width:86px; position:-webkit-sticky; position:sticky; left:0; z-index:2; background:#0e0e0d; padding-left:8px !important; padding-right:4px; } "
            "th:last-child, td:last-child { min-width:42px; max-width:42px; width:42px; position:-webkit-sticky; position:sticky; right:0; z-index:2; background:#0e0e0d; text-align:right; padding-left:4px; padding-right:8px; box-shadow:-2px 0 0 0 #0e0e0d; } "
            "tbody tr:hover td:first-child, tbody tr:hover td:last-child { background:#0e0e0d !important; } "
        )
    _detail_headers = (
        '<th>Driver</th><th>Qualifying</th><th>Finish</th>'
        f'<th>Pos <span style="font-size:13px;line-height:1">\u00b1</span></th><th>Overtakes</th>'
        f'<th>FL (prob)</th><th>DOTD (prob)</th>{sprint_header}<th>Pts</th>'
    )
    _sortable_table("dtbl", _detail_headers, table_rows,
                    '<col>' * (8 + has_sprint_col),
                    36 + len(breakdown_df) * 41 + 24, _detail_css)