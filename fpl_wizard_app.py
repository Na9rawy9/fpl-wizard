import os
import ast
import html
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# CONFIG
# ============================================================
SEASON = "2025-2026"
TOUR_NAME = "Premier League"
NUM_GW = 38

st.set_page_config(page_title="FPL Wizard Mobile", layout="wide", initial_sidebar_state="collapsed")

# ============================================================
# UTILS
# ============================================================
def first_col(df, names):
    return next((x for x in names if x in df.columns), None)

def num_col(df, names):
    c = first_col(df, names)
    return pd.Series(0.0, index=df.index) if c is None else pd.to_numeric(df[c], errors="coerce").fillna(0)

# ============================================================
# DATA LOADING & AGGREGATION
# ============================================================
@st.cache_data(show_spinner="Loading and building player datasets (this will be cached)...")
def load_data():
    repo_root = os.getcwd()
    target_data_dir = None
    
    for dirpath, _, filenames in os.walk(repo_root):
        if "players.csv" in filenames and "teams.csv" in filenames:
            target_data_dir = dirpath
            break
            
    if not target_data_dir:
        return pd.DataFrame(), pd.DataFrame()
        
    players = pd.read_csv(os.path.join(target_data_dir, "players.csv"))
    # Ensure no duplicates in players manifest
    if "player_id" in players.columns:
        players = players.drop_duplicates("player_id")
        
    teams = pd.read_csv(os.path.join(target_data_dir, "teams.csv"))

    players["team_code_internal"] = players["team_code"] if "team_code" in players.columns else players["team"]
    
    pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD', '1': 'GKP', '2': 'DEF', '3': 'MID', '4': 'FWD', '1.0': 'GKP', '2.0': 'DEF', '3.0': 'MID', '4.0': 'FWD'}
    if "element_type" in players.columns:
        players["position"] = players["element_type"].astype(str).map(pos_map).fillna("UNK")
    elif "position" in players.columns:
        players["position"] = players["position"].astype(str).str.upper()
        players["position"] = players["position"].replace({'GOALKEEPER': 'GKP', 'DEFENDER': 'DEF', 'MIDFIELDER': 'MID', 'FORWARD': 'FWD'})

    tour_dir = None
    for dirpath, dirnames, _ in os.walk(repo_root):
        if TOUR_NAME in dirnames or f"GW1" in dirnames:
            if "GW1" in dirnames:
                tour_dir = dirpath
            else:
                tour_dir = os.path.join(dirpath, TOUR_NAME)
            break
            
    ml, pl, gl = [], [], []
    
    if tour_dir:
        for gw in range(1, NUM_GW + 1):
            p = os.path.join(tour_dir, f"GW{gw}")
            m_path = os.path.join(p, "matches.csv")
            pm_path = os.path.join(p, "playermatchstats.csv")
            g_path = os.path.join(p, "player_gameweek_stats.csv")
    
            if os.path.exists(m_path):
                m = pd.read_csv(m_path)
                m["gameweek"] = gw
                ml.append(m)
            if os.path.exists(pm_path):
                pm = pd.read_csv(pm_path)
                pm["gameweek"] = gw
                pl.append(pm)
            if os.path.exists(g_path):
                g = pd.read_csv(g_path)
                g["gameweek"] = gw
                gl.append(g)

    matches = pd.concat(ml, ignore_index=True) if ml else pd.DataFrame()
    pms = pd.concat(pl, ignore_index=True) if pl else pd.DataFrame()
    pgw = pd.concat(gl, ignore_index=True) if gl else pd.DataFrame()

    if pms.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Highly robust deduplication to prevent Matches > 38 bug caused by dirty data
    if "match_id" in pms.columns:
        pms = pms.drop_duplicates(subset=["player_id", "match_id"])
    else:
        pms = pms.drop_duplicates()
        
    pgw = pgw.drop_duplicates(subset=["id", "gameweek"]) if "id" in pgw.columns else pgw.drop_duplicates()

    ids = pms.groupby("player_id")["minutes_played"].sum()
    ids = ids[ids > 0].index
    players = players[players.player_id.isin(ids)].copy()
    pms = pms[pms.player_id.isin(ids)]
    pgw = pgw[pgw.id.isin(ids)]

    strength_col = "elo" if "elo" in teams.columns else "strength"
    if not teams.empty and strength_col in teams.columns:
        lo, hi = teams[strength_col].min(), teams[strength_col].max()
    else:
        lo, hi = 1, 5
        
    tbc = teams.set_index("code") if not teams.empty else pd.DataFrame()

    def to_fdr(strength):
        if pd.isna(strength) or hi == lo: return 3
        return max(1, min(5, 1 + int((strength - lo) / max(hi - lo, 1e-9) * 5)))

    stat_cols = [x for x in [
        "goalsscored","goals_scored","goals","assists","cleansheets","clean_sheets","bonus",
        "expected_goals","xg","expected_assists","xa","expected_goal_involvements","xgi",
        "expected_goals_conceded","xgc"
    ] if x in pgw.columns]

    defcons_col = first_col(pgw, ["defensivecontribution", "defensive_contribution", "defensivecontributions"])
    if defcons_col:
        if defcons_col not in stat_cols:
            stat_cols.append(defcons_col)

    pos_cols = ["player_id", "position"]
    pos_df = players[pos_cols].rename(columns={"player_id": "id"})
    pgw = pgw.merge(pos_df, on="id", how="left")

    def calc_defcon_pts(row):
        val = pd.to_numeric(row[defcons_col], errors="coerce")
        if pd.isna(val) or val <= 0:
            return 0
        pos = str(row.get("position", "")).upper()
        is_gkp_def = "DEF" in pos or "GKP" in pos
        is_mid_fwd = "MID" in pos or "FWD" in pos
        if is_gkp_def:
            return 2 if val >= 10 else 0
        elif is_mid_fwd:
            return 2 if val >= 12 else 0
        return 0

    if defcons_col in pgw.columns:
        pgw["defcons_pts"] = pgw.apply(calc_defcon_pts, axis=1)
        stat_cols.append("defcons_pts")

    season_stats = pgw.groupby("id")[stat_cols].sum().reset_index().rename(columns={"id": "player_id"})

    snap = pgw[pgw.gameweek == pgw.gameweek.max()].copy().rename(columns={"id": "player_id"})
    price_df = snap[["player_id", "now_cost"]] if "now_cost" in snap.columns else pd.DataFrame(columns=["player_id", "now_cost"])

    players = players.merge(price_df, on="player_id", how="left")
    players = players.merge(season_stats, on="player_id", how="left")

    price_s = pd.to_numeric(players.now_cost, errors="coerce").fillna(0)
    if price_s.max() > 30:
        players["price_m"] = price_s / 10.0
    else:
        players["price_m"] = price_s

    records = []
    cs_col_name = first_col(pgw, ["cleansheets", "clean_sheets"])
    goals_col_name = first_col(pgw, ["goalsscored", "goals_scored", "goals"])
    assists_col_name = first_col(pgw, ["assists"])
    xg_col_name = first_col(pgw, ["expected_goals", "xg"])
    xa_col_name = first_col(pgw, ["expected_assists", "xa"])
    xgi_col_name = first_col(pgw, ["expected_goal_involvements", "expected_goals_involvements", "xgi"])
    xgc_col_name = first_col(pgw, ["expected_goals_conceded", "xgc"])

    max_gw = int(pgw.gameweek.max()) if not pgw.empty else NUM_GW

    for gw in range(1, max_gw + 1):
        ms = matches[matches.gameweek == gw]
        ps = pms[pms.gameweek == gw]
        gs = pgw[pgw.gameweek == gw]

        for _, p in players.iterrows():
            player_matches = ps[ps.player_id == p.player_id]
            player_gw_row = gs[gs.id == p.player_id]
            pos = p.get("position", "")

            total_mins = player_matches.minutes_played.sum() if not player_matches.empty else 0

            if not player_gw_row.empty:
                pts = player_gw_row.event_points.sum() if "event_points" in player_gw_row.columns else 0
                cs_val = pd.to_numeric(player_gw_row[cs_col_name], errors="coerce").sum() if cs_col_name else 0
                defcon_val = pd.to_numeric(player_gw_row["defcons_pts"], errors="coerce").sum() if "defcons_pts" in player_gw_row.columns else 0
                gw_goals = pd.to_numeric(player_gw_row[goals_col_name], errors="coerce").sum() if goals_col_name else 0
                gw_assists = pd.to_numeric(player_gw_row[assists_col_name], errors="coerce").sum() if assists_col_name else 0

                raw_xg = pd.to_numeric(player_gw_row[xg_col_name], errors="coerce").sum() if xg_col_name else 0
                raw_xa = pd.to_numeric(player_gw_row[xa_col_name], errors="coerce").sum() if xa_col_name else 0
                raw_xgi = pd.to_numeric(player_gw_row[xgi_col_name], errors="coerce").sum() if xgi_col_name else 0
                raw_xgc = pd.to_numeric(player_gw_row[xgc_col_name], errors="coerce").sum() if xgc_col_name else 0
                raw_defcons = pd.to_numeric(player_gw_row[defcons_col], errors="coerce").sum() if defcons_col else 0

                if pos in ["GKP", "DEF"]:
                    cs_fpl_points = cs_val * 4
                elif pos == "MID":
                    cs_fpl_points = cs_val * 1
                else:
                    cs_fpl_points = 0
            else:
                pts = np.nan
                cs_val = 0
                defcon_val = 0
                gw_goals = 0
                gw_assists = 0
                cs_fpl_points = 0
                raw_xg = 0
                raw_xa = 0
                raw_xgi = 0
                raw_xgc = 0
                raw_defcons = 0

            # More robust match counting to prevent exceeding 38 matches
            if "match_id" in player_matches.columns:
                actual_matches = player_matches[player_matches.minutes_played > 0].match_id.nunique()
            else:
                actual_matches = (player_matches.minutes_played > 0).sum()
                
            starts_count = 0
            if "start_min" in player_matches.columns and not player_matches.empty:
                starts_count = ((player_matches.start_min == 0) & (player_matches.minutes_played > 0)).sum()

    grid = pd.DataFrame(records).sort_values(["player_id", "gameweek"]).drop_duplicates(["player_id", "gameweek"])
    if not grid.empty:
        grid["team_short"] = grid.team_code_internal.map(teams.set_index("code").short_name)
    return grid, players

@st.cache_data(show_spinner=False)
def metrics(grid, players, base_grid_for_participation=None):
    if grid.empty:
        return pd.DataFrame()
        
    m = grid.groupby("player_id").agg(
        web_name=("web_name", "first"),
        price_m=("price_m", "first"),
        position=("position", "first"),
        team_short=("team_short", "first"),
        total_minutes=("total_minutes", "sum"),
        matches_played=("actual_matches", "sum"),
        participation_gws=("actual_matches", lambda x: int((x > 0).sum())),
        possible_gws=("gameweek", "count"),
        starts=("starts_count", "sum"),
        total_points=("points", lambda x: pd.to_numeric(x, errors="coerce").fillna(0).sum())
    ).reset_index()

    # Cap matches_played at 38 (just in case of external data anomalies)
    m["matches_played"] = m["matches_played"].clip(upper=38)

    def calc_robust_stability(group):
        pts = group['points'].dropna()
        if len(pts) < 2:
            return 100.0 if (len(pts) == 1 and pts.iloc[0] > 0) else 0.0
        q1 = np.percentile(pts, 25)
        q3 = np.percentile(pts, 75)
        iqr = q3 - q1
        robust_std = iqr / 1.349
        mean = pts.mean()
        if mean <= 0:
            return 0.0
        return 100.0 * (mean / (mean + robust_std))
        
    stability_df = grid[grid.total_minutes > 0].groupby("player_id").apply(calc_robust_stability).rename("stability").reset_index()
    m = m.merge(stability_df, on="player_id", how="left").fillna({"stability": 0.0})

    m["avg_minutes_per_match"] = np.where(m.matches_played > 0, m.total_minutes / m.matches_played, 0)
    m["avg_points_per_match"] = np.where(m.matches_played > 0, m.total_points / m.matches_played, 0)
    
    m["participation_percent"] = np.where(m.possible_gws > 0, m.participation_gws / m.possible_gws * 100, 0)
    m["start_percent"] = np.where(m.matches_played > 0, m.starts / m.matches_played * 100, 0)

    cs_hits = grid[(grid.cs > 0) & (grid.total_minutes > 0)].groupby("player_id").size().rename("cs_hits")
    defcon_hits = grid[(grid.defcons_pts > 0) & (grid.total_minutes > 0)].groupby("player_id").size().rename("defcon_hits")

    m = m.merge(cs_hits, on="player_id", how="left").fillna({"cs_hits": 0})
    m = m.merge(defcon_hits, on="player_id", how="left").fillna({"defcon_hits": 0})

    m["pct_cs"] = np.where((m.matches_played > 0) & (m.position.isin(["GKP", "DEF"])), m.cs_hits / m.matches_played * 100, 0)
    m["pct_defcon"] = np.where(m.matches_played > 0, m.defcon_hits / m.matches_played * 100, 0)

    cs_defcons_points = grid.groupby("player_id").apply(lambda group: group["cs_fpl_points"].sum() + group["defcons_pts"].sum(), include_groups=False).rename("cs_defcons_points").reset_index()
    m = m.merge(cs_defcons_points, on="player_id", how="left").fillna({"cs_defcons_points": 0})

    total_cs_pts = grid.groupby("player_id")["cs_fpl_points"].sum().rename("clean_sheets_pts").reset_index()
    m = m.merge(total_cs_pts, on="player_id", how="left").fillna({"clean_sheets_pts": 0})

    s = players[["player_id"]].copy()
    s["goals"] = num_col(players, ["goalsscored", "goals_scored", "goals"])
    s["assists_sort"] = num_col(players, ["assists"])
    s["defcons"] = num_col(players, ["defcons_pts"]) if "defcons_pts" in players.columns else pd.Series(0.0, index=s.index)
    s["bonus_points"] = num_col(players, ["bonus"])
    s["xg"] = num_col(players, ["expected_goals", "xg"])
    s["xa"] = num_col(players, ["expected_assists", "xa"])
    s["xgi"] = num_col(players, ["expected_goal_involvements", "expected_goals_involvements", "xgi"])
    s["xgc"] = num_col(players, ["expected_goals_conceded", "xgc"])
    s["defcons_raw"] = num_col(players, ["defensivecontribution", "defensive_contribution", "defensivecontributions"])

    m = m.merge(s, on="player_id", how="left")
    m["goal_assists"] = m.goals + m.assists_sort

    m["xg_per_90"] = np.where(m.total_minutes > 0, (m.xg / m.total_minutes) * 90, 0)
    m["xa_per_90"] = np.where(m.total_minutes > 0, (m.xa / m.total_minutes) * 90, 0)
    m["xgi_per_90"] = np.where(m.total_minutes > 0, (m.xgi / m.total_minutes) * 90, 0)
    m["xgc_per_90"] = np.where(m.total_minutes > 0, (m.xgc / m.total_minutes) * 90, 0)
    m["defcon_per_90"] = np.where(m.total_minutes > 0, (m.defcons_raw / m.total_minutes) * 90, 0)

    return m

def add_delivery_consistency(metric_df, grid_df, target, window):
    if grid_df.empty:
        return metric_df.assign(delivery_consistency=0)
    output = []
    for pid, group in grid_df.groupby("player_id"):
        points = group.sort_values("gameweek").set_index("gameweek")["points"].reindex(range(1, int(grid_df.gameweek.max()) + 1)).fillna(0)
        rolls = points.rolling(window).mean().dropna()
        consistency = ((rolls >= target).mean() * 100) if len(rolls) else 0
        output.append((pid, consistency))
    consistency_df = pd.DataFrame(output, columns=["player_id", "delivery_consistency"])
    return metric_df.merge(consistency_df, on="player_id", how="left").fillna({"delivery_consistency": 0})


# ============================================================
# TRANSPOSED HTML TABLE RENDERER (Mobile Optimized)
# ============================================================
def render_transposed_html(df, metrics_def):
    out = []
    out.append("<div class='table-container'><table id='stats-table'>")
    
    out.append("<thead><tr>")
    out.append("<th class='sticky-tl'>Metric</th>")
    for _, r in df.iterrows():
        out.append(f"<th class='sticky-t'>{html.escape(r['web_name'])}</th>")
    out.append("</tr></thead>")
    
    out.append("<tbody>")
    for idx, (m_key, m_label, m_type) in enumerate(metrics_def):
        out.append("<tr>")
        # Add sortCols onclick event to the metric header cell
        out.append(f"<td class='sticky-l sortable-row' onclick='sortCols(this, {idx})' title='Click to sort columns by {m_label}'>{html.escape(m_label)} &#8597;</td>")
        
        for _, r in df.iterrows():
            val = r[m_key]
            
            if pd.isna(val):
                disp = "-"
            elif m_type == "string":
                disp = html.escape(str(val))
            elif m_type == "cost":
                disp = f"&pound;{float(val):.1f}m"
            elif m_type == "int":
                disp = f"{int(val)}"
            elif m_type == "float2":
                disp = f"{float(val):.2f}"
            elif m_type == "percent":
                v = float(val)
                disp = f"<div class='prog-bg'><div class='prog-fg' style='width: {min(100, max(0, v))}%;'></div></div><div>{v:.1f}%</div>"
            else:
                disp = str(val)
                
            out.append(f"<td>{disp}</td>")
        out.append("</tr>")
        
    out.append("</tbody></table></div>")
    
    css = """
    <style>
    :root { --border-color: #353946; --bg-dark: #0e1117; --bg-head: #1b1f29; }
    html, body { margin:0; padding:0; background:var(--bg-dark); color:white; font-family:sans-serif; }
    
    /* Highly responsive scroll container */
    .table-container { 
        width: 100%; 
        height: 75vh; 
        overflow: auto; 
        -webkit-overflow-scrolling: touch; 
        box-sizing: border-box; 
    }
    
    /* Scrollbar styling */
    .table-container::-webkit-scrollbar { width: 6px; height: 6px; }
    .table-container::-webkit-scrollbar-thumb { background: #5b6270; border-radius: 3px; }
    
    table { border-collapse: separate; border-spacing: 0; min-width: 100%; table-layout: fixed; }
    th, td { border-right: 1px solid var(--border-color); border-bottom: 1px solid var(--border-color); padding: 8px 4px; box-sizing: border-box; vertical-align: middle; text-align: center; }
    
    /* Header Row (Player Names) */
    th.sticky-t { position: sticky; top: 0; z-index: 10; background: var(--bg-head); border-top: 1px solid var(--border-color); font-size: 13px; font-weight: bold; width: 23vw; min-width: 80px; max-width: 120px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    
    /* Top Left Corner */
    th.sticky-tl { position: sticky; top: 0; left: 0; z-index: 12; background: var(--bg-head); border-top: 1px solid var(--border-color); width: 26vw; min-width: 95px; max-width: 140px; }
    
    /* Left Column (Metric Names) */
    td.sticky-l { position: sticky; left: 0; z-index: 9; background: var(--bg-head); text-align: left; font-size: 12px; font-weight: 600; width: 26vw; min-width: 95px; max-width: 140px; color: #f0f2f6; }
    
    /* Clickable Sortable Row Hover State */
    td.sortable-row { cursor: pointer; transition: background 0.2s, color 0.2s; }
    td.sortable-row:hover { background: #2c3140; color: #fff; }
    
    /* Data Cells */
    td { font-size: 12px; color: #eee; width: 23vw; min-width: 80px; max-width: 120px; overflow: hidden; }
    
    /* Progress Bars CSS */
    .prog-bg { width: 100%; background: #333; height: 6px; border-radius: 3px; margin-bottom: 4px; overflow: hidden; }
    .prog-fg { height: 100%; background: #009c00; border-radius: 3px; }
    </style>
    """
    
    js = """
    <script>
    var sortDirs = {};
    function sortCols(triggerCell, rowIdx) {
        var table = triggerCell.closest('table');
        var tbody = table.querySelector('tbody');
        var thead = table.querySelector('thead');
        var targetRow = tbody.rows[rowIdx];
        
        var dirKey = 'row_' + rowIdx;
        var newDir = (sortDirs[dirKey] === 'desc') ? 'asc' : 'desc';
        sortDirs[dirKey] = newDir;
        
        var colData = [];
        
        // Loop through all cells in the clicked row, EXCEPT the first cell (the metric name)
        for (var i = 1; i < targetRow.cells.length; i++) {
            var cell = targetRow.cells[i];
            var rawVal = cell.innerText || cell.textContent;
            
            // Clean value for number parsing
            var cleanVal = rawVal.replace(/[^0-9.-]/g, '');
            var num = parseFloat(cleanVal);
            
            colData.push({
                index: i,
                raw: rawVal,
                num: num,
                isNum: !isNaN(num) && cleanVal.length > 0
            });
        }
        
        // Sort the array of column objects
        colData.sort(function(a, b) {
            if (a.isNum && b.isNum) {
                return (newDir === 'asc') ? (a.num - b.num) : (b.num - a.num);
            } else {
                return (newDir === 'asc') ? a.raw.localeCompare(b.raw) : b.raw.localeCompare(a.raw);
            }
        });
        
        // Apply the new order to ALL rows (Header + Body)
        var allRows = Array.from(thead.rows).concat(Array.from(tbody.rows));
        
        allRows.forEach(function(row) {
            var cells = Array.from(row.cells);
            
            // Remove all cells except the very first one (index 0)
            while (row.cells.length > 1) {
                row.deleteCell(1);
            }
            
            // Re-append the stored cells in the newly sorted order
            colData.forEach(function(col) {
                row.appendChild(cells[col.index]);
            });
        });
    }
    </script>
    """
    
    return css + "".join(out) + js


# ============================================================
# APP UI & LAYOUT
# ============================================================

master_grid, players = load_data()

if master_grid.empty:
    st.error("No data found for the current season. Please check your data folders.")
    st.stop()

# Filter integration explicitly pushed to the top of the UI
st.title("🧙‍♂️ FPL Wizard")
st.markdown("Mobile-Optimized Player Statistics")

search_query = st.text_input("🔍 Search Player", "", help="Filter players by typing their name.")

colA, colB = st.columns(2)
with colA:
    selected_positions = st.pills("Positions", ["GKP", "DEF", "MID", "FWD"], selection_mode="multi", default=[])
with colB:
    fdr_choices = ["FDR = 1", "FDR = 2", "FDR = 3", "FDR = 4", "FDR = 5"]
    fdr_selection = st.pills("FDR Filter", fdr_choices, selection_mode="multi", default=[])

price_min = players['price_m'].min() if not players.empty else 3.5
price_max = players['price_m'].max() if not players.empty else 15.0
price_range = st.slider("Price Range (£m)", float(price_min), float(price_max), (float(price_min), float(price_max)), 0.1, format="£%.1fm")

with st.expander("⚙️ Advanced Filters (Teams, Mins, Participation, Targets)"):
    teams_list = sorted(master_grid.team_short.dropna().unique())
    selected_teams = st.multiselect("Teams", teams_list, default=[])
    
    use_mins = st.checkbox("Average Minutes Per Match", False)
    avg_mins = st.slider("Min Avg Minutes", 0, 90, 60, disabled=not use_mins)
    
    use_part = st.checkbox("Participation Across GWs", False)
    part_pct = st.slider("Min Participation %", 0, 100, 75, disabled=not use_part)
    
    use_starts = st.checkbox("Starts / Participations", False)
    starts_pct = st.slider("Min Starts %", 0, 100, 75, disabled=not use_starts)
    
    st.divider()
    target = st.number_input("Delivery Target Points", -10, 30, 4, 1)
    window = st.number_input("Consistency Window (GWs)", 2, 10, 3, 1)


# --- APPLY GLOBAL FDR FILTER ---
if fdr_selection:
    target_fdrs = [int(f.split("= ")[1]) for f in fdr_selection]
    grid = master_grid[master_grid.fdr.isin(target_fdrs)].copy()
else:
    grid = master_grid.copy()

if grid.empty:
    st.warning("No matches available for the selected FDR filter.")
    st.stop()

# Generate Metrics from (Filtered) Grid
m = metrics(grid, players, base_grid_for_participation=grid)

deliv_mask = (grid.points >= target) & (grid.total_minutes > 0)
deliveries = grid[deliv_mask].groupby("player_id").size().rename("deliveries")
pts_delivered = grid[deliv_mask].groupby("player_id")["points"].sum().rename("pts_delivered")
m = m.merge(deliveries, on="player_id", how="left").fillna({"deliveries": 0})
m = m.merge(pts_delivered, on="player_id", how="left").fillna({"pts_delivered": 0})
m["pct_delivery"] = np.where(m.matches_played > 0, m.deliveries / m.matches_played * 100, 0)
m["xdelivery"] = np.where(m.total_points != 0, m.pts_delivered / m.total_points * 100, 0)

m = add_delivery_consistency(m, grid, target, int(window))

# Apply General Filters
filtered_m = m.copy()

if search_query:
    filtered_m = filtered_m[filtered_m.web_name.str.contains(search_query, case=False, na=False)]

if selected_positions:
    filtered_m = filtered_m[filtered_m.position.isin(selected_positions)]

if selected_teams:
    filtered_m = filtered_m[filtered_m.team_short.isin(selected_teams)]

filtered_m = filtered_m[(filtered_m.price_m >= price_range[0]) & (filtered_m.price_m <= price_range[1])]

if use_mins:
    filtered_m = filtered_m[filtered_m.avg_minutes_per_match >= avg_mins]
if use_part:
    filtered_m = filtered_m[filtered_m.participation_percent >= part_pct]
if use_starts:
    filtered_m = filtered_m[filtered_m.start_percent >= starts_pct]

if filtered_m.empty:
    st.warning("No players match these filters.")
    st.stop()

# --- DEFINE METRICS FOR TRANSPOSED TABLE ---
metrics_def = [
    ("team_short", "Team", "string"),
    ("position", "Pos", "string"),
    ("price_m", "Cost", "cost"),
    ("total_points", "Total Pts", "int"),
    ("matches_played", "Matches", "int"),
    ("starts", "Starts", "int"),
    ("avg_points_per_match", "Avg Pts/M", "float2"),
    ("avg_minutes_per_match", "Avg Mins", "int"),
    ("pct_delivery", "Delivery %", "percent"),
    ("delivery_consistency", "Consistency %", "percent"),
    ("stability", "Stability %", "percent"),
    ("xg_per_90", "xG/90", "float2"),
    ("xa_per_90", "xA/90", "float2"),
    ("xgi_per_90", "xGI/90", "float2"),
    ("xgc_per_90", "xGC/90", "float2"),
    ("defcon_per_90", "DefCon/90", "float2"),
    ("pct_cs", "CS %", "percent"),
    ("pct_defcon", "DefCon %", "percent"),
    ("xdelivery", "xDelivery %", "percent"),
    ("goals", "Goals", "int"),
    ("assists_sort", "Assists", "int"),
    ("bonus_points", "Bonus", "int"),
    ("clean_sheets_pts", "CS Pts", "int"),
    ("defcons", "DefCons", "int"),
    ("cs_defcons_points", "CS+DC Pts", "int")
]

# Initially sort by Total Points internally
ordered_m = filtered_m.sort_values("total_points", ascending=False)

st.markdown("### 📊 Player Statistics")
st.caption("👈 Tap any metric name on the left to re-sort the players instantly!")

html_payload = render_transposed_html(ordered_m, metrics_def)
components.html(html_payload, height=750, scrolling=False)

# Add Glossary for mobile users below the table
with st.expander("📖 Metric Glossary & Definitions"):
    st.markdown("""
    * **Delivery %**: The percentage of played matches where the player successfully reached your specified Delivery Target.
    * **Consistency %**: Evaluates the player on a rolling Gameweek window. If they consistently average the target points across every window, they score highly here.
    * **Stability %**: A measure of a player's baseline reliability. Calculated using the Interquartile Range to explicitly ignore extreme outliers (like a random 15-point haul on a normally 2-point player). A higher percentage means highly predictable returns.
    * **xG / xA / xGC**: Expected Goals, Expected Assists, and Expected Goals Conceded (scaled per 90 minutes).
    * **DefCon**: Defensive Contributions (e.g. saves, clearances, recoveries).
    """)