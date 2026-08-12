import os
import ast
import re
import html
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px

# ============================================================
# CONFIG
# ============================================================
REPO_ROOT = r"."
SEASON = "2025-2026"
TOUR_NAME = "Premier League"
NUM_GW = 38

st.set_page_config(page_title="FPL Wizard", layout="wide", initial_sidebar_state="expanded")

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
@st.cache_data(show_spinner="Loading and building large player datasets (this will be cached)...")
def load_data():
    root = os.path.join(REPO_ROOT, "data", SEASON)
    
    if not os.path.exists(os.path.join(root, "players.csv")):
        return pd.DataFrame(), pd.DataFrame()
        
    players = pd.read_csv(os.path.join(root, "players.csv"))
    teams = pd.read_csv(os.path.join(root, "teams.csv"))

    players["team_code_internal"] = players["team_code"] if "team_code" in players.columns else players["team"]
    
    pos_map = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD', '1': 'GKP', '2': 'DEF', '3': 'MID', '4': 'FWD', '1.0': 'GKP', '2.0': 'DEF', '3.0': 'MID', '4.0': 'FWD'}
    if "element_type" in players.columns:
        players["position"] = players["element_type"].astype(str).map(pos_map).fillna("UNK")
    elif "position" in players.columns:
        players["position"] = players["position"].astype(str).str.upper()
        players["position"] = players["position"].replace({'GOALKEEPER': 'GKP', 'DEFENDER': 'DEF', 'MIDFIELDER': 'MID', 'FORWARD': 'FWD'})

    ml, pl, gl = [], [], []
    for gw in range(1, NUM_GW + 1):
        p = os.path.join(root, "By Tournament", TOUR_NAME, f"GW{gw}")
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

    ids = pms.groupby("player_id")["minutes_played"].sum()
    ids = ids[ids > 0].index
    players = players[players.player_id.isin(ids)].copy()
    pms = pms[pms.player_id.isin(ids)]
    pgw = pgw[pgw.id.isin(ids)]

    strength_col = "elo" if "elo" in teams.columns else "strength"
    lo, hi = teams[strength_col].min(), teams[strength_col].max()
    tbc = teams.set_index("code")

    def to_fdr(strength):
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

            actual_matches = (player_matches.minutes_played > 0).sum() if not player_matches.empty else 0
            starts_count = 0
            if "start_min" in player_matches.columns and not player_matches.empty:
                starts_count = ((player_matches.start_min == 0) & (player_matches.minutes_played > 0)).sum()
                if starts_count > 0:
                    status = ""
                elif total_mins > 0:
                    status = "SUB"
                else:
                    status = "DNP"
            else:
                if total_mins > 0:
                    status = "SUB"
                else:
                    status = "DNP"

            opps, flags, fdrs, per_match_mins = [], [], [], []
            is_blank, is_double = False, False

            if not player_matches.empty:
                matches_to_process = player_matches.merge(ms, on="match_id", how="inner")
            else:
                matches_to_process = ms[(ms.home_team == p.team_code_internal) | (ms.away_team == p.team_code_internal)]

            matches_to_process = matches_to_process.drop_duplicates("match_id")

            if len(matches_to_process) == 0:
                is_blank = True
            elif len(matches_to_process) > 1:
                is_double = True

            for _, mr in matches_to_process.iterrows():
                h_code = mr.home_team
                a_code = mr.away_team
                opp_code = None
                is_home = None
                
                if h_code == p.team_code_internal:
                    opp_code = a_code
                    is_home = True
                elif a_code == p.team_code_internal:
                    opp_code = h_code
                    is_home = False
                else:
                    if not player_gw_row.empty and "opponent_team" in player_gw_row.columns:
                        opp_ids_raw = player_gw_row["opponent_team"].values[0]
                        opp_ids = []
                        if isinstance(opp_ids_raw, str):
                            try:
                                parsed = ast.literal_eval(opp_ids_raw)
                                opp_ids = parsed if isinstance(parsed, list) else [int(opp_ids_raw)]
                            except:
                                opp_ids = [int(opp_ids_raw)]
                        elif pd.notna(opp_ids_raw):
                            opp_ids = [int(opp_ids_raw)]
                            
                        opp_team_codes = teams[teams["id"].isin(opp_ids)]["code"].tolist()
                        
                        if h_code in opp_team_codes:
                            opp_code = h_code
                            is_home = False
                        elif a_code in opp_team_codes:
                            opp_code = a_code
                            is_home = True
                
                if opp_code is not None and opp_code in tbc.index:
                    opp_team = tbc.loc[opp_code]
                    opp_name = opp_team.short_name
                    opp_str = opp_team.elo if "elo" in opp_team.index else opp_team.strength
                    fdr = to_fdr(opp_str)
                else:
                    fdr = 3
                    if opp_code is None:
                        h_name = tbc.loc[h_code, "short_name"] if h_code in tbc.index else str(h_code)
                        a_name = tbc.loc[a_code, "short_name"] if a_code in tbc.index else str(a_code)
                        opp_name = f"{h_name} v {a_name}"
                        is_home = None
                    else:
                        opp_name = "UNK"

                opps.append(opp_name)
                flags.append("H" if is_home is True else ("A" if is_home is False else ""))
                fdrs.append(fdr)

                if "minutes_played" in mr:
                    per_match_mins.append(str(int(mr.minutes_played)))
                else:
                    per_match_mins.append("0")

            if opps:
                opp_parts = []
                for o, h in zip(opps, flags):
                    opp_parts.append(f"{o} ({h})" if h else o)
                opp_text = " / ".join(opp_parts)
                avg_fdr = int(round(np.mean(fdrs)))
                mins_text = "&prime; / ".join(per_match_mins) + "&prime;" if len(opp_parts) > 1 else f"{per_match_mins[0]}&prime;"
            else:
                opp_text = ""
                avg_fdr = np.nan
                mins_text = "0&prime;"

            records.append(dict(
                player_id=p.player_id, web_name=p.web_name, gameweek=gw,
                total_minutes=total_mins, mins_text=mins_text,
                actual_matches=actual_matches, starts_count=starts_count,
                points=pts, status=status, opp=opp_text, fdr=avg_fdr, price_m=p.price_m,
                position=pos, team_code_internal=p.team_code_internal,
                cs=cs_val, defcons_pts=defcon_val, cs_fpl_points=cs_fpl_points,
                goals=gw_goals, assists=gw_assists,
                xg=raw_xg, xa=raw_xa, xgi=raw_xgi, xgc=raw_xgc, defcons_raw=raw_defcons,
                is_blank=is_blank, is_double=is_double
            ))

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

    # Robust Stability Calculation: Interquartile Range (IQR) based standard deviation
    # This ignores outliers (like a random 15-point haul) and accurately reflects regular baseline output
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

    # Clean Sheet % logic: Zero out for attackers (MID/FWD) as they receive minimal/no CS points
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


def make_fixtures_html(df, ids, dark, green, yellow, orange, delivery_mode, delivery_target, metric_dict, metric_name, syncGroup="main"):
    names = df[["player_id", "web_name"]].drop_duplicates("player_id").set_index("player_id").web_name.to_dict()
    gws = sorted(df.gameweek.unique())
    lookup = df.drop_duplicates(["player_id", "gameweek"]).set_index(["player_id", "gameweek"]).to_dict("index")

    def perf_bucket(pts, mins, status, dark, green, yellow, orange):
        if status == "DNP" or mins == 0: return 0
        if pd.isna(pts): return 5
        if pts >= dark: return 1
        if pts >= green: return 2
        if pts >= yellow: return 3
        if pts >= orange: return 4
        return 5

    def cell_text(opp, fdr):
        if not opp: return "&mdash;"
        text = html.escape(str(opp))
        return text if pd.isna(fdr) else f"{text} {'&#9733;'*int(fdr)}"

    table_id = f"{syncGroup}Fixtures"
    out = [f"<div class='wrap'><table id='{table_id}' class='fixtures-table'><thead><tr>"]
    
    out.append("<th class='p'><div class='sth'>")
    out.append(f"<div class='nm' onclick=\"sortSyncTables(this.closest('th'), 'fixtures', 'alpha', '{syncGroup}')\" style='cursor:pointer' title='Sort by Name'>Player &#8597;</div>")
    out.append(f"<div class='mt' onclick=\"sortSyncTables(this.closest('th'), 'fixtures', 'metric', '{syncGroup}')\" style='cursor:pointer' title='Sort by {html.escape(metric_name)}'>{html.escape(metric_name)} &#8597;</div>")
    out.append("</div></th>")

    out.extend([f"<th>GW{x}</th>" for x in gws])
    out.append("</tr></thead><tbody>")

    colors = {0: "#000000", 1: "#006400", 2: "#009c00", 3: "#b4b400", 4: "#d47a00", 5: "#b00000"}

    for pid in ids:
        pname = html.escape(str(names.get(pid, "Unknown")))
        mval = html.escape(str(metric_dict.get(pid, "")))
        out.append(f"<tr data-pid='{pid}'><td class='n'><div class='std'><div class='nm'>{pname}</div><div class='mt'>{mval}</div></div></td>")
        for gw in gws:
            x = lookup.get((pid, gw))
            if x is None:
                out.append("<td class='c' style='background:#20232b;'></td>")
                continue
            
            bucket = perf_bucket(x["points"], int(x["total_minutes"]), x["status"], dark, green, yellow, orange)
            bg = "#000000" if bucket == 0 else ("#009c00" if delivery_mode and not pd.isna(x["points"]) and x["points"] >= delivery_target else ("#b00000" if delivery_mode else colors[bucket]))
            pts = "" if pd.isna(x["points"]) else int(x["points"])
            
            status_val = x['status']
            if status_val == "SUB":
                status_disp = ", <span style='color: #000000; background-color: #ffffff; padding: 0px 3px; border-radius: 3px; font-weight: 900;'>SUB</span>"
            elif status_val:
                status_disp = f", {status_val}"
            else:
                status_disp = ""

            stats_line = f"{x['mins_text']} / {pts} pts{status_disp}"

            tr_html = "<div class='tr-badge'>DGW</div>" if x.get("is_double", False) else ("<div class='tr-badge'>BGW</div>" if x.get("is_blank", False) else "")
            
            br_tags = []
            if x.get("cs", 0) > 0 and x.get("total_minutes", 0) > 0: br_tags.append("CS")
            if x.get("defcons_pts", 0) > 0 and x.get("total_minutes", 0) > 0: br_tags.append("DC")
            br_html = f"<div class='br-badge'>{' / '.join(br_tags)}</div>" if br_tags else ""

            bl_tags = []
            goals = int(x.get("goals", 0))
            assists = int(x.get("assists", 0))
            if goals > 0 and x.get("total_minutes", 0) > 0: 
                bl_tags.append(f"<span>{'&#9917;'*goals}</span>")
            if assists > 0 and x.get("total_minutes", 0) > 0: 
                bl_tags.append(f"<span>{'&#127919;'*assists}</span>")
                
            bl_html = f"<div class='bl-badge'>{''.join(bl_tags)}</div>" if bl_tags else ""

            out.append(f"<td class='c' style='background:{bg};'><div class='c-in'><span class='a'>{cell_text(x['opp'], x['fdr'])}</span><span class='b'>{stats_line}</span></div>{tr_html}{bl_html}{br_html}</td>")
        out.append("</tr>")
    out.append("</tbody></table></div>")
    
    css = """
    <style>
    html, body { margin:0; padding:0; width:100%; height:100%; overflow:hidden; background:#0e1117; color:white; font-family:Arial; }
    .wrap { width:100%; height:100vh; overflow:auto; position:relative; box-sizing:border-box; padding-bottom:20px; }
    table { border-collapse:separate; border-spacing:0; min-width:100%; width:max-content; table-layout:fixed; }
    th, td { border:1px solid #353946; border-top:none; border-left:none; padding:0; margin:0; box-sizing:border-box; }
    thead th { border-top:1px solid #353946; border-bottom:2px solid #353946; }
    .fixtures-table th { position:sticky; top:0; z-index:5; background:#1b1f29; min-width:145px; width:145px; height:42px; text-align:center;}
    .fixtures-table th.p { position:sticky; left:0; z-index:7; background:#1b1f29; min-width:260px; width:260px; border-right:2px solid #353946; }
    .fixtures-table td.n { position:sticky; left:0; z-index:4; background:#0e1117; min-width:260px; width:260px; border-right:2px solid #353946; }
    .sth { display:flex; height:100%; align-items:center; }
    .std { display:flex; height:100%; align-items:center; }
    .nm { flex:1; min-width:160px; padding-left:12px; text-align:left; font-weight:600; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-size:13px; color:#f0f2f6; }
    .mt { width:100px; min-width:100px; text-align:center; border-left:1px solid #353946; font-size:12px; color:#ccc; font-weight:bold; height:100%; display:flex; align-items:center; justify-content:center; }
    td.c { min-width:145px; height:52px; position:relative; padding:3px; }
    .c-in { display:flex; flex-direction:column; height:100%; justify-content:center; align-items:center; }
    .a { font-weight:bold; font-size:11px; white-space:nowrap; }
    .b { font-size:10px; color:#eee; white-space:nowrap; }
    sup { font-size:8px; position:relative; top:-.3em; margin-left:2px; }
    .br-badge { position:absolute; bottom:2px; right:3px; font-size:8px; font-weight:bold; color:rgba(255,255,255,0.8); line-height:1; }
    .bl-badge { position:absolute; bottom:2px; left:3px; font-size:8px; font-weight:bold; color:rgba(255,255,255,0.8); line-height:1; display:flex; gap:3px; }
    .tr-badge { position:absolute; top:2px; right:3px; font-size:8px; font-weight:bold; color:#000; background-color:#fff; padding:0 3px; border-radius:2px; line-height:1.2; }
    .wrap::-webkit-scrollbar { width:14px; height:14px; }
    .wrap::-webkit-scrollbar-thumb { background:#5b6270; border-radius:8px; border:3px solid #0e1117; }
    </style>
    """
    js = """
    <script>
    var sortDirs = {};
    function sortSyncTables(thElement, tableType, subType, syncGroup) {
        var colIndex = thElement.cellIndex;
        var table = document.getElementById(syncGroup + 'Fixtures');
        if (!table) return;
        var tbody = table.getElementsByTagName('tbody')[0];
        var rows = Array.from(tbody.getElementsByTagName('tr'));
        var colIdentifier = thElement.innerText.trim();
        var dirKey = syncGroup + '_' + colIdentifier;
        var newDir = (sortDirs[dirKey] === 'desc') ? 'asc' : 'desc';
        sortDirs[dirKey] = newDir;
        rows.sort(function(a, b) {
            var cellX = a.getElementsByTagName('td')[colIndex];
            var cellY = b.getElementsByTagName('td')[colIndex];
            var valX = cellX ? cellX.querySelector('.nm') ? cellX.querySelector(subType === 'alpha' ? '.nm' : '.mt').innerText.trim() : cellX.innerText : '';
            var valY = cellY ? cellY.querySelector('.nm') ? cellY.querySelector(subType === 'alpha' ? '.nm' : '.mt').innerText.trim() : cellY.innerText : '';
            valX = valX.replace(/[^0-9.-]/g, '');
            valY = valY.replace(/[^0-9.-]/g, '');
            var numX = parseFloat(valX);
            var numY = parseFloat(valY);
            var isNumX = !isNaN(numX);
            var isNumY = !isNaN(numY);
            if (isNumX && isNumY) {
                return (newDir === 'asc') ? (numX - numY) : (numY - numX);
            } else {
                return (newDir === 'asc') ? valX.localeCompare(valY) : valY.localeCompare(valX);
            }
        });
        rows.forEach(function(row) { tbody.appendChild(row); });
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

st.title("🧙‍♂️ FPL Wizard")
st.markdown("Advanced FPL Projections, Statistics & Comparisons")

# --- SIDEBAR FILTERS ---
with st.sidebar:
    st.header("🎯 Core Filters")
    
    search_query = st.text_input("🔍 Search Player", "", help="Filter players by typing their name.")
    
    selected_positions = st.pills(
        "Positions", 
        ["GKP", "DEF", "MID", "FWD"], 
        selection_mode="multi",
        default=[],
        help="Leave unselected to view all positions."
    )

    fdr_choices = ["FDR = 1", "FDR = 2", "FDR = 3", "FDR = 4", "FDR = 5"]
    fdr_selection = st.pills(
        "FDR Filter",
        fdr_choices,
        selection_mode="multi",
        default=[],
        help="Leave unselected to view Season-long data, or pick specific fixture difficulties (1-5)."
    )

    price_min = players['price_m'].min() if not players.empty else 3.5
    price_max = players['price_m'].max() if not players.empty else 15.0
    price_range = st.slider(
        "Price Range (£m)", 
        min_value=float(price_min), 
        max_value=float(price_max), 
        value=(float(price_min), float(price_max)), 
        step=0.1,
        format="£%.1fm"
    )
    
    teams_list = sorted(master_grid.team_short.dropna().unique())
    selected_teams = st.multiselect(
        "Teams", 
        teams_list, 
        default=[],
        help="Leave empty to display all teams."
    )
    
    st.divider()
    
    with st.expander("⏱️ Playing Time Thresholds", expanded=False):
        use_mins = st.checkbox("Average Minutes Per Match", False, help="Filter out players who average fewer minutes per match than the selected threshold.")
        avg_mins = st.slider("Min Avg Minutes", 0, 90, 60, disabled=not use_mins)
        
        use_part = st.checkbox("Participation Across GWs", False, help="Filter out players who have appeared in fewer than the selected percentage of the Gameweeks (calculated based on available matches matching your active FDR filter).")
        part_pct = st.slider("Min Participation %", 0, 100, 75, disabled=not use_part)
        
        use_starts = st.checkbox("Starts / Participations", False, help="Filter out players who start fewer than the selected percentage of their played matches.")
        starts_pct = st.slider("Min Starts %", 0, 100, 75, disabled=not use_starts)
        
    with st.expander("🎯 Delivery Targets", expanded=False):
        target = st.number_input("Delivery Target Points", -10, 30, 4, 1, help="Points needed in a Gameweek or rolling window to count as a delivery.")
        window = st.number_input("Consistency Window (GWs)", 2, 10, 3, 1, help="Number of rolling Gameweeks (e.g. 3 evaluates GW1-3, GW2-4...).")


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
    st.warning("No players match these filters. (Note: Using the Participation % filter alongside an FDR filter requires the player to have participated in enough games OF THAT SPECIFIC FDR).")
    st.stop()

# --- MAIN TABS ---
tab_stat, tab_compare, tab_fixtures = st.tabs([
    "📊 Advanced Statistics", 
    "⚖️ Player Radar Comparison",
    "📅 Player Fixtures"
])

shared_column_config = {
    "web_name": st.column_config.TextColumn("Player Name", width="medium"),
    "team_short": st.column_config.TextColumn("Team", width="small", help="Players Team"),
    "position": st.column_config.TextColumn("Pos", width="small", help="Player Position (GKP, DEF, MID, FWD)"),
    "price_m": st.column_config.NumberColumn("Cost", format="£%.1fm", width="small", help="Current FPL Cost"),
    "pct_delivery": st.column_config.ProgressColumn(
        "Delivery %", format="%.1f%%", min_value=0, max_value=100, 
        help="Percentage of played matches where the player reached the delivery target."
    ),
    "delivery_consistency": st.column_config.ProgressColumn(
        "Consistency %", format="%.1f%%", min_value=0, max_value=100, 
        help="Percentage of rolling gameweek windows where the delivery target was met."
    ),
    "stability": st.column_config.ProgressColumn(
        "Stability %", format="%.1f%%", min_value=0, max_value=100, 
        help="Measures how stable the player's points are, ignoring extreme outliers (based on Interquartile Range)."
    ),
    "total_points": st.column_config.NumberColumn("Total Pts", format="%d", help="Total FPL points scored by the player."),
    "avg_points_per_match": st.column_config.NumberColumn("Avg Pts/M", format="%.2f", help="Average points scored per match played."),
    "xg_per_90": st.column_config.NumberColumn("xG/90", format="%.2f", help="Expected Goals per 90 minutes played."),
    "xa_per_90": st.column_config.NumberColumn("xA/90", format="%.2f", help="Expected Assists per 90 minutes played."),
    "xgi_per_90": st.column_config.NumberColumn("xGI/90", format="%.2f", help="Expected Goal Involvements (xG + xA) per 90 minutes played."),
    "xgc_per_90": st.column_config.NumberColumn("xGC/90", format="%.2f", help="Expected Goals Conceded per 90 minutes played."),
    "defcon_per_90": st.column_config.NumberColumn("DefCon/90", format="%.2f", help="Raw Defensive Contributions per 90 minutes played."),
    "pct_cs": st.column_config.NumberColumn("CS %", format="%.1f%%", help="Percentage of played matches where the player kept a Clean Sheet."),
    "pct_defcon": st.column_config.NumberColumn("DefCon %", format="%.1f%%", help="Percentage of played matches where the player registered a Defensive Contribution."),
    "xdelivery": st.column_config.NumberColumn("xDelivery %", format="%.1f%%", help="Expected Delivery proportion of total points coming from deliveries."),
    "matches_played": st.column_config.NumberColumn("Matches", format="%d", help="Total games participated in."),
    "starts": st.column_config.NumberColumn("Starts", format="%d", help="Total games played as a starter."),
    "avg_minutes_per_match": st.column_config.NumberColumn("Avg Mins", format="%d", help="Average minutes played per match."),
    "goals": st.column_config.NumberColumn("Goals", format="%d", help="Total goals scored."),
    "assists_sort": st.column_config.NumberColumn("Assists", format="%d", help="Total assists provided."),
    "bonus_points": st.column_config.NumberColumn("Bonus", format="%d", help="Total FPL bonus points accumulated."),
    "clean_sheets_pts": st.column_config.NumberColumn("CS Pts", format="%d", help="Total FPL points earned from Clean Sheets."),
    "defcons": st.column_config.NumberColumn("DefCons", format="%d", help="Total defensive contribution points earned."),
    "cs_defcons_points": st.column_config.NumberColumn("CS+DC Pts", format="%d", help="Total points earned specifically from Clean Sheets and DefCons."),
}

with tab_stat:
    st.subheader(f"📊 Player Statistics ({len(filtered_m)} players)")
    
    display_cols = [
        "web_name", "team_short", "position", "price_m", 
        "pct_delivery", "delivery_consistency", "stability",
        "total_points", "matches_played", "starts", 
        "avg_points_per_match", "avg_minutes_per_match", 
        "xg_per_90", "xa_per_90", "xgi_per_90", "xgc_per_90", "defcon_per_90", 
        "pct_cs", "pct_defcon", "xdelivery", "goals", "assists_sort", "bonus_points", 
        "clean_sheets_pts", "defcons", "cs_defcons_points"
    ]
    
    st.dataframe(
        filtered_m[display_cols].sort_values("total_points", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config=shared_column_config,
        height=600
    )


with tab_compare:
    st.subheader("⚖️ Visual Player Comparison")
    
    compare_players = st.multiselect(
        "Select up to 6 players to compare", 
        options=filtered_m.web_name.tolist(),
        default=filtered_m.sort_values("total_points", ascending=False).web_name.head(3).tolist() if not filtered_m.empty else [],
        max_selections=6,
        help="Use this dropdown to build a list of players to compare on the radar charts below."
    )
    
    if compare_players:
        comp_df = filtered_m[filtered_m.web_name.isin(compare_players)].copy()
        
        col1, col2 = st.columns(2)
        
        def plot_radar(df, metrics_list, title, name_map):
            norm_df = df[['web_name'] + metrics_list].copy()
            for col in metrics_list:
                max_val = filtered_m[col].max() if filtered_m[col].max() > 0 else 1
                norm_df[col] = norm_df[col] / max_val
                
            melt_df = norm_df.melt(id_vars=['web_name'], value_vars=metrics_list, var_name='Metric', value_name='Score')
            melt_df['Metric'] = melt_df['Metric'].map(name_map)
            
            fig = px.line_polar(
                melt_df, 
                r='Score', 
                theta='Metric', 
                color='web_name', 
                line_close=True,
                markers=True,
                title=title,
                template="plotly_dark"
            )
            # Position the legend completely to the left
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
                legend=dict(
                    title_text="",
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="right",
                    x=-0.1
                ),
                margin=dict(l=120, r=20, t=40, b=20)
            )
            return fig

        with col1:
            off_metrics = ['avg_points_per_match', 'pct_delivery', 'delivery_consistency', 'xg_per_90', 'xa_per_90']
            off_map = {
                'avg_points_per_match': 'Avg Pts/Match', 
                'pct_delivery': '% Delivery', 
                'delivery_consistency': '% Consistency', 
                'xg_per_90': 'xG / 90',
                'xa_per_90': 'xA / 90'
            }
            st.plotly_chart(plot_radar(comp_df, off_metrics, "Offensive Output", off_map), use_container_width=True)

        with col2:
            def_metrics = ['avg_points_per_match', 'pct_delivery', 'delivery_consistency', 'pct_cs', 'defcon_per_90']
            def_map = {
                'avg_points_per_match': 'Avg Pts/Match', 
                'pct_delivery': '% Delivery', 
                'delivery_consistency': '% Consistency', 
                'pct_cs': '% Clean Sheets',
                'defcon_per_90': 'DefCon / 90'
            }
            st.plotly_chart(plot_radar(comp_df, def_metrics, "Defensive Output", def_map), use_container_width=True)

    else:
        st.info("Select players above to compare them visually.")


with tab_fixtures:
    st.subheader("📅 Player Fixtures")
    st.markdown("Displays opponents color-coded by FDR with exact goals, assists, CS, and DGW/BGW markers.")
    
    # Note: Fixtures Tab explicitly uses the full un-filtered master_grid so that
    # filtering by FDR in the sidebar doesn't arbitrarily wipe out columns in the fixture UI.
    fix_grid = master_grid[master_grid.player_id.isin(filtered_m.player_id)].copy()
    
    if not fix_grid.empty:
        colA, colB, colC = st.columns([2, 1, 1])
        with colA:
            sort_metric_name = st.selectbox(
                "Metric Column:",
                [
                    "Total Points", "Cost", "Average Points Per Match", "Delivery %", "Consistency %", "Stability %",
                    "xG/90", "xA/90", "xGI/90", "xGC/90", "DefCon/90", "CS %", "DefCon %",
                    "xDelivery %", "Matches", "Starts", "Avg Mins", "Goals", "Assists", 
                    "Bonus", "CS Pts", "DefCons", "CS+DC Pts"
                ],
                help="Select which metric appears next to the player's name in the fixtures table."
            )
        with colB:
            delivery_mode = st.checkbox("Delivery Coloring Scheme", False, help="Colors cells green when the player reaches the delivery target, red otherwise.")
        
        metric_col_map = {
            "Cost": "price_m",
            "Total Points": "total_points",
            "Average Points Per Match": "avg_points_per_match",
            "Delivery %": "pct_delivery",
            "Consistency %": "delivery_consistency",
            "Stability %": "stability",
            "xG/90": "xg_per_90",
            "xA/90": "xa_per_90",
            "xGI/90": "xgi_per_90",
            "xGC/90": "xgc_per_90",
            "DefCon/90": "defcon_per_90",
            "CS %": "pct_cs",
            "DefCon %": "pct_defcon",
            "xDelivery %": "xdelivery",
            "Matches": "matches_played",
            "Starts": "starts",
            "Avg Mins": "avg_minutes_per_match",
            "Goals": "goals",
            "Assists": "assists_sort",
            "Bonus": "bonus_points",
            "CS Pts": "clean_sheets_pts",
            "DefCons": "defcons",
            "CS+DC Pts": "cs_defcons_points"
        }
        
        metric_dict = {}
        target_col = metric_col_map[sort_metric_name]
        for _, r in filtered_m.iterrows():
            val = r[target_col]
            if sort_metric_name == "Cost":
                metric_dict[r.player_id] = f"{val:.1f}m"
            elif sort_metric_name in ["Delivery %", "Consistency %", "Stability %", "CS %", "DefCon %", "xDelivery %"]:
                metric_dict[r.player_id] = f"{val:.1f}%"
            elif sort_metric_name in ["Total Points", "Matches", "Starts", "Avg Mins", "Goals", "Assists", "Bonus", "CS Pts", "DefCons", "CS+DC Pts"]:
                metric_dict[r.player_id] = f"{int(val)}"
            else:
                metric_dict[r.player_id] = f"{val:.2f}"
                
        ordered_ids = filtered_m.sort_values(target_col, ascending=False).player_id.tolist()
        
        # scrolling=False inside components.html forces the internal `.wrap` CSS container to handle horizontal scrolling gracefully
        html_code = make_fixtures_html(
            df=fix_grid, 
            ids=ordered_ids, 
            dark=10, 
            green=6, 
            yellow=4, 
            orange=1, 
            delivery_mode=delivery_mode, 
            delivery_target=target, 
            metric_dict=metric_dict, 
            metric_name=sort_metric_name
        )
        
        components.html(html_code, height=720, scrolling=False)
        
        st.markdown("""
        ---
        **📊 Fixtures Legend & Acronyms:**
        * **Opponent (H/A) ★**: The opposing team, whether it was Home (H) or Away (A), and the Fixture Difficulty Rating (FDR) represented by stars (1 to 5).
        * **Minutes / Pts**: The top line inside a cell displays the minutes played and the exact FPL points earned.
        * **SUB / DNP**: Player was a substitute (SUB) or Did Not Play (DNP).
        * **DGW / BGW**: Double Gameweek (two matches in one GW) / Blank Gameweek (no match).
        * **CS / DC**: Clean Sheet (CS) maintained / Defensive Contribution (DC) recorded.
        * **⚽ / 🎯**: Each ball represents a single goal scored; each dartboard represents a single assist.
        """)
    else:
        st.info("No fixtures available for the selected players.")