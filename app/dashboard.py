"""Streamlit dashboard for the F1 fantasy team optimiser."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


from app.optimiser.optimiser import optimiser

REPORTS_DIR = Path("reports")

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

st.set_page_config(page_title="F1 Fantasy Optimiser", layout="wide")

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

/* h2/h3 also get serif — covers subheader() and ### Recommended team */
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
.main .block-container { padding-top: 0.5rem; }

/* Tabs — smaller, right-aligned */
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
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>F1 Fantasy<br/>Team Optimiser</h1>", unsafe_allow_html=True)



# formats driver/constructor IDs as display names e.g. max_verstappen -> Max Verstappen
def format_name(id_str):
    return id_str.replace("_", " ").title()


tab1, tab2 = st.tabs(["Team Picker", "Model Performance"])


# team picker tab
with tab1:
    pred_path = REPORTS_DIR / "predictions_latest.json"

    if not pred_path.exists():
        st.info("No predictions available yet.")
        st.stop()

    data = json.loads(pred_path.read_text())
    driver_team = {d["driver_id"]: d["constructor_id"] for d in data["drivers"] if "constructor_id" in d}

    st.subheader(f"Season {data['season']} - Round {data['round']}: {data['circuit']}")
    st.caption(f"Last updated {data['generated_at'][:10]}")

    driver_df = pd.DataFrame(data["drivers"]).rename(columns={"expected_points": "expected_fantasy_points"})
    constructor_df = pd.DataFrame(data["constructors"]).rename(columns={"expected_points": "expected_fantasy_points"})
    prices_df = pd.DataFrame(
        [{"asset_id": d["driver_id"], "price": d["price"]} for d in data["drivers"]] +
        [{"asset_id": c["constructor_id"], "price": c["price"]} for c in data["constructors"]]
    )
    prices_index = prices_df.set_index("asset_id")["price"]

    col_left, col_spacer, col_right = st.columns([1, 0.1, 2])

    with col_left:
        budget = st.number_input("Budget (£M)", min_value=0.0, max_value=200.0, value=100.0, step=0.1)
        error_slot = st.empty()

        has_current_team = st.checkbox("Enter current team")
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
            current_constructors = st.pills(
                "Current constructors",
                options=[c["constructor_id"] for c in sorted(data["constructors"], key=lambda x: x["price"], reverse=True)],
                format_func=format_name,
                selection_mode="multi",
                default=[],
            )
            if len(current_drivers) > 5:
                error_slot.error("Select at most 5 drivers.")
                st.stop()
            if len(current_constructors) > 2:
                error_slot.error("Select at most 2 constructors.")
                st.stop()
            team_cost = sum(float(prices_index.get(i, 0)) for i in current_drivers + current_constructors)
            if team_cost > 0:
                remaining = budget - team_cost
                if remaining < 0:
                    error_slot.error(f"Team costs £{team_cost:.1f}M - £{abs(remaining):.1f}M over budget.")
                    st.stop()
                st.progress(min(team_cost / budget, 1.0), text=f"£{team_cost:.1f}M spent - £{remaining:.1f}M remaining")

            free_transfers = st.radio("Free transfers", [2, 3], horizontal=True)

        state = None
        if current_drivers or current_constructors:
            team_cost = sum(float(prices_index.get(i, 0)) for i in current_drivers + current_constructors)
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
            pts = float(driver_df.set_index("driver_id")["expected_fantasy_points"][d])
            price = float(prices_index[d])
            doubled = d == team["doubled_driver"]
            display_pts = pts * 2 if doubled else pts
            driver_rows.append({"Driver": format_name(d) + (" x2" if doubled else ""), "Pts": f"{display_pts:.1f}", "Price": f"£{price:.1f}M"})
            total_points += display_pts
            total_cost += price

        constructor_rows = []
        for c in team["constructors"]:
            pts = float(constructor_df.set_index("constructor_id")["expected_fantasy_points"][c])
            price = float(prices_index[c])
            constructor_rows.append({"Constructor": format_name(c), "Pts": f"{pts:.1f}", "Price": f"£{price:.1f}M"})
            total_points += pts
            total_cost += price

        def _tr(cells, tag="td"):
            return "<tr>" + "".join(f"<{tag}>{c}</{tag}>" for c in cells) + "</tr>"

        table_html = """
        <style>
        .team-table { width:100%; border-collapse:collapse; font-family:'DM Sans',sans-serif; font-size:13px; color:#f7f6f3; }
        .team-table th { color:rgba(247,246,243,0.35); font-size:10px; font-weight:400; letter-spacing:0.1em; text-transform:uppercase; padding:6px 12px 8px 12px; text-align:left; border-bottom:1px solid rgba(247,246,243,0.12); }
        .team-table td { padding:10px 12px; border-bottom:1px solid rgba(247,246,243,0.06); }
        .team-table tr:last-child td { border-bottom:1px solid rgba(247,246,243,0.12); }
        .team-table col.name { width:55%; }
        .team-table col.stat { width:22%; }
        .team-table .gap td { height:24px; border:none; }
        </style>
        <table class="team-table">
          <colgroup><col class="name"><col class="stat"><col class="stat"></colgroup>
          <thead>""" + _tr(["Driver", "Pts", "Price"], "th") + """</thead>
          <tbody>""" + "".join(_tr([r["Driver"], r["Pts"], r["Price"]]) for r in driver_rows) + """
            <tr class="gap"><td colspan="3"></td></tr>
          </thead>
          <thead>""" + _tr(["Constructor", "Pts", "Price"], "th") + """</thead>
          <tbody>""" + "".join(_tr([r["Constructor"], r["Pts"], r["Price"]]) for r in constructor_rows) + """
          </tbody>
        </table>
        <div style="height:1.5rem"></div>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        extra = team["transfer_penalty"]
        penalty_note = f' <span style="font-size:1rem;color:rgba(247,246,243,0.45);font-family:\'DM Sans\',sans-serif;font-weight:300">(-{extra * 10}pt, {extra} extra transfer{"s" if extra != 1 else ""})</span>' if state and extra > 0 else ""
        st.markdown(f'<p style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:rgba(247,246,243,0.4);margin-bottom:4px">Projected points</p><p style="font-family:\'Cormorant Garamond\',serif;font-size:2.4rem;font-weight:300;color:#f7f6f3;margin:0 0 1rem 0">{total_points:.1f}{penalty_note}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:10px;letter-spacing:0.1em;text-transform:uppercase;color:rgba(247,246,243,0.4);margin-bottom:4px">Total cost</p><p style="font-family:\'Cormorant Garamond\',serif;font-size:2.4rem;font-weight:300;color:#f7f6f3;margin:0 0 1rem 0">£{total_cost:.1f}M <span style="font-size:1rem;color:rgba(247,246,243,0.45);font-family:\'DM Sans\',sans-serif;font-weight:300">/ £{budget:.1f}M</span></p>', unsafe_allow_html=True)

    with col_right:
        # per-chart x ranges with 0 aligned at the same relative position in both charts
        d_left = abs(min(driver_df["expected_fantasy_points"].min(), 0)) * 1.1
        d_right = driver_df["expected_fantasy_points"].max() * 1.05
        zero_fraction = d_left / (d_left + d_right) if (d_left + d_right) > 0 else 0

        c_right = constructor_df["expected_fantasy_points"].max() * 1.05
        c_left = (zero_fraction * c_right / (1 - zero_fraction)) if zero_fraction > 0 else 0

        driver_x_range = [-d_left, d_right]
        constructor_x_range = [-c_left, c_right]

        # drivers bar chart - sorted by expected points, selected team highlighted in red with bold label
        driver_chart = driver_df[["driver_id", "expected_fantasy_points", "price"]].assign(
            name=lambda df: df["driver_id"].apply(lambda x: f"<b>{format_name(x)}</b>" if x in selected_ids else format_name(x)),
            color=lambda df: df["driver_id"].apply(lambda x: TEAM_COLORS.get(driver_team.get(x, ""), "#f7f6f3") if x in selected_ids else "#3a3a38"),
        ).sort_values("expected_fantasy_points")

        fig_drivers = px.bar(
            driver_chart,
            x="expected_fantasy_points",
            y="name",
            orientation="h",
            hover_data={"price": ":.1f", "color": False, "name": False},
            labels={"expected_fantasy_points": "Expected Points", "name": "", "price": "Price (£M)"},
        )
        fig_drivers.update_traces(marker_color=driver_chart["color"].tolist())
        fig_drivers.update_layout(
            showlegend=False, height=500, margin=dict(l=130, r=0, t=36, b=0),
            paper_bgcolor="#0e0e0d", plot_bgcolor="#0e0e0d",
            font=dict(family="DM Sans", color="#f7f6f3"),
            title=dict(text="Drivers", font=dict(size=11, color="rgba(247,246,243,0.4)", family="DM Sans"), x=0),
            xaxis=dict(gridcolor="rgba(247,246,243,0.08)", range=driver_x_range,
                       title_font=dict(family="DM Sans", size=11, color="rgba(247,246,243,0.6)"),
                       tickfont=dict(family="DM Sans", size=11, color="rgba(247,246,243,0.6)")),
            yaxis=dict(automargin=False, tickfont=dict(family="DM Sans", size=11, color="rgba(247,246,243,0.6)")),
        )
        st.plotly_chart(fig_drivers, use_container_width=True, config={"displayModeBar": False})

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
            hover_data={"price": ":.1f", "color": False, "name": False},
            labels={"expected_fantasy_points": "Expected Points", "name": "", "price": "Price (£M)"},
        )
        fig_constructors.update_traces(marker_color=constructor_chart["color"].tolist())
        fig_constructors.update_layout(
            showlegend=False, height=300, margin=dict(l=130, r=0, t=36, b=0),
            paper_bgcolor="#0e0e0d", plot_bgcolor="#0e0e0d",
            font=dict(family="DM Sans", color="#f7f6f3"),
            title=dict(text="Constructors", font=dict(size=11, color="rgba(247,246,243,0.4)", family="DM Sans"), x=0),
            xaxis=dict(gridcolor="rgba(247,246,243,0.08)", range=constructor_x_range,
                       title_font=dict(family="DM Sans", size=11, color="rgba(247,246,243,0.6)"),
                       tickfont=dict(family="DM Sans", size=11, color="rgba(247,246,243,0.6)")),
            yaxis=dict(automargin=False, tickfont=dict(family="DM Sans", size=11, color="rgba(247,246,243,0.6)")),
        )
        st.plotly_chart(fig_constructors, use_container_width=True, config={"displayModeBar": False})


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

    summary = (
        all_data.groupby("season")[["model", "oracle"]]
        .sum()
        .assign(pct_of_oracle=lambda df: (df["model"] / df["oracle"] * 100).round(1))
        .rename(columns={"model": "Model", "oracle": "Oracle", "pct_of_oracle": "% of Oracle"})
    )
    summary.index.name = "Season"

    st.subheader("Season summary")
    st.dataframe(
        summary,
        use_container_width=True,
        column_config={
            "Season": st.column_config.NumberColumn(width="small"),
            "Model": st.column_config.NumberColumn(width="medium"),
            "Oracle": st.column_config.NumberColumn(width="medium"),
            "% of Oracle": st.column_config.NumberColumn(width="medium"),
        },
    )

    st.subheader("Cumulative points by strategy")
    season_options = sorted(all_data["season"].unique())
    selected_season = st.selectbox("Season", season_options, index=len(season_options) - 1)

    season_data = all_data[all_data["season"] == selected_season].copy()
    season_data["Model"] = season_data["model"].cumsum()
    season_data["Oracle"] = season_data["oracle"].cumsum()

    fig2 = px.line(
        season_data.melt(id_vars="round", value_vars=["Model", "Oracle"], var_name="Strategy", value_name="Cumulative Points"),
        x="round",
        y="Cumulative Points",
        color="Strategy",
        color_discrete_map={"Model": "#c8401a", "Oracle": "rgba(247,246,243,0.4)"},
        labels={"round": "Round"},
        markers=True,
    )
    fig2.update_layout(
        paper_bgcolor="#0e0e0d", plot_bgcolor="#0e0e0d",
        font=dict(family="DM Sans", color="#f7f6f3"),
        xaxis=dict(gridcolor="rgba(247,246,243,0.08)"),
        yaxis=dict(gridcolor="rgba(247,246,243,0.08)"),
    )
    st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # per-round % of oracle achieved
    season_data["pct_of_oracle"] = (season_data["model"] / season_data["oracle"] * 100).round(1)
    max_round = all_data["round"].max()
    fig3 = px.bar(
        season_data,
        x="round",
        y="pct_of_oracle",
        hover_data={"model": True, "oracle": True},
        labels={"round": "Round", "pct_of_oracle": "% of Oracle", "model": "Model pts", "oracle": "Oracle pts"},
        title="% of oracle achieved per round",
    )
    fig3.update_traces(marker_color="#c8401a")
    fig3.add_hline(y=season_data["pct_of_oracle"].mean(), line_dash="dash", line_color="rgba(247,246,243,0.3)",
                   annotation_text=f"avg {season_data['pct_of_oracle'].mean():.1f}%", annotation_position="top right")
    fig3.add_hline(y=100, line_dash="solid", line_color="rgba(247,246,243,0.08)", line_width=1)
    fig3.update_yaxes(range=[0, 105])
    fig3.update_xaxes(range=[0.5, max_round + 0.5])
    fig3.update_layout(
        margin=dict(t=40),
        paper_bgcolor="#0e0e0d", plot_bgcolor="#0e0e0d",
        font=dict(family="DM Sans", color="#f7f6f3"),
        xaxis=dict(gridcolor="rgba(247,246,243,0.08)"),
        yaxis=dict(gridcolor="rgba(247,246,243,0.08)"),
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})