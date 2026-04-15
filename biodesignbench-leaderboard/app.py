"""BioDesignBench Leaderboard — Gradio App for HuggingFace Spaces

Evaluating LLM Agents on Protein Design via MCP Tools
Romero Lab, Duke University

Tabs:
  1. Overall Leaderboard
  2. Taxonomy Breakdown
  3. Component Analysis
  4. Benchmark vs User
  5. Submit (new submission form)
  6. Status & Admin (password-protected pipeline control)
  7. About
"""

import json
import os
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go

ADMIN_PASSWORD = os.environ.get("BDB_ADMIN_PASSWORD", "biodesignbench2026")


# ═══════════════════════════════════════════════════════════════════
#  Configuration — change these when deploying
# ═══════════════════════════════════════════════════════════════════

PAPER_URL = "#"
GITHUB_URL = "#"
HF_URL = "#"


# ═══════════════════════════════════════════════════════════════════
#  Taxonomy & scoring constants (2 × 5 design matrix)
# ═══════════════════════════════════════════════════════════════════

APPROACHES = ["de_novo", "redesign"]
APPROACH_LABELS = {
    "de_novo": "De Novo Design",
    "redesign": "Redesign",
}
SUBJECTS = ["antibody", "binder", "enzyme", "scaffold", "fluorescent_protein"]
SUBJECT_LABELS = {
    "antibody": "Antibody",
    "binder": "Binder",
    "enzyme": "Enzyme",
    "scaffold": "Scaffold",
    "fluorescent_protein": "Fluorescent Prot.",
}
# 9 valid cells (rd × binder is empty in current task set)
VALID_CELLS = {
    "de_novo": {"antibody", "binder", "enzyme", "scaffold", "fluorescent_protein"},
    "redesign": {"antibody", "enzyme", "scaffold", "fluorescent_protein"},
}
N_TASKS_PER_CELL = {
    ("de_novo", "antibody"): 4,
    ("de_novo", "binder"): 19,
    ("de_novo", "enzyme"): 2,
    ("de_novo", "scaffold"): 21,
    ("de_novo", "fluorescent_protein"): 1,
    ("redesign", "antibody"): 5,
    ("redesign", "enzyme"): 10,
    ("redesign", "scaffold"): 4,
    ("redesign", "fluorescent_protein"): 10,
}
COMPONENTS = [
    "approach",
    "orchestration",
    "quality",
    "feasibility",
    "novelty",
    "diversity",
]
COMP_MAX = {
    "approach": 20,
    "orchestration": 15,
    "quality": 35,
    "feasibility": 15,
    "novelty": 5,
    "diversity": 10,
}
TYPE_STYLE = {
    "llm": {"icon": "", "bg": "#ffffff", "tag": ""},
    "hardcoded": {"icon": "\U0001f527", "bg": "#f0f0f0", "tag": "baseline"},
    "human_expert": {
        "icon": "\U0001f468\u200d\U0001f52c",
        "bg": "#ebf4ff",
        "tag": "baseline",
    },
    "human_oracle": {"icon": "\U0001f4c4", "bg": "#fefcbf", "tag": "baseline"},
    # Backward-compat alias for older JSON files
    "oracle": {"icon": "\U0001f4c4", "bg": "#fefcbf", "tag": "baseline"},
}


# ═══════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════


def load_data() -> dict:
    path = Path(__file__).parent / "leaderboard_data.json"
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; }
.gr-padded { padding: 0 !important; }

/* Force light appearance for all inline-styled HTML content */
.dark .gradio-container {
  --body-background-fill: #f7fafc !important;
  --block-background-fill: #ffffff !important;
  --body-text-color: #1a202c !important;
  --block-label-text-color: #1a202c !important;
  --input-background-fill: #ffffff !important;
  --border-color-primary: #e2e8f0 !important;
  --color-accent-soft: rgba(49,130,206,0.15) !important;
  --neutral-50: #f7fafc !important;
  --neutral-100: #edf2f7 !important;
  --neutral-200: #e2e8f0 !important;
  --neutral-700: #4a5568 !important;
  --neutral-800: #2d3748 !important;
  color: #1a202c !important;
  background: #f7fafc !important;
}
.dark .tabs { background: #ffffff !important; }
.dark .tab-nav button { color: #2d3748 !important; }
.dark .tab-nav button.selected {
  color: #0f172a !important;
  border-color: #3182ce !important;
}
.dark .block { background: #ffffff !important; }
.dark label, .dark .label-wrap { color: #2d3748 !important; }
.dark input, .dark textarea, .dark select {
  background: #ffffff !important;
  color: #1a202c !important;
  border-color: #e2e8f0 !important;
}
.dark .accordion { background: #ffffff !important; }
.dark .accordion > .label-wrap { color: #2d3748 !important; }
"""

# Force light mode on page load
FORCE_LIGHT_JS = """
() => {
  document.querySelector('body').classList.remove('dark');
  const obs = new MutationObserver(() => {
    document.querySelector('body').classList.remove('dark');
  });
  obs.observe(document.body, {attributes: true, attributeFilter: ['class']});
  setTimeout(() => obs.disconnect(), 5000);
}
"""


# ═══════════════════════════════════════════════════════════════════
#  Plotly layout helper
# ═══════════════════════════════════════════════════════════════════


def _base_layout(**overrides) -> dict:
    """Shared Plotly layout defaults, with per-chart overrides."""
    base = dict(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            family="system-ui, -apple-system, sans-serif", size=12, color="#2d3748"
        ),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════
#  HTML builders
# ═══════════════════════════════════════════════════════════════════


def build_header(last_updated: str, n_entries: int) -> str:
    btn = (
        "display:inline-block;padding:0.45rem 1.1rem;border-radius:8px;"
        "text-decoration:none;font-size:0.82rem;font-weight:600;"
        "transition:opacity 0.15s"
    )
    return f"""
    <div style="background:#ffffff;border:1px solid #e2e8f0;
                padding:2.2rem 2rem 1.8rem;text-align:center;
                border-radius:16px;margin-bottom:0.8rem;
                box-shadow:0 1px 4px rgba(0,0,0,0.04)">
      <p style="margin:0 0 0.3rem;font-size:0.75rem;font-weight:700;
                letter-spacing:0.12em;text-transform:uppercase;
                color:#3182ce">Romero Lab &middot; Duke University</p>
      <h1 style="font-size:2rem;margin:0;font-weight:800;color:#0f172a;
                  letter-spacing:-0.02em">
        \U0001f9ec BioDesignBench</h1>
      <p style="color:#0f172a;margin:0.6rem 0 0.2rem;font-size:1.1rem;
                font-weight:600;line-height:1.4">
        Can LLM agents orchestrate stochastic protein-design pipelines?</p>
      <p style="color:#64748b;margin:0.2rem 0 0;font-size:0.95rem;
                font-weight:400;font-style:italic;max-width:680px;
                margin-left:auto;margin-right:auto;line-height:1.5">
        Top-tier agents now surpass a deterministic pipeline &mdash;
        but invoke evaluation tools at only <strong>14% of expert depth</strong>.
        Guidance rescues coverage, not depth.</p>
      <div style="margin-top:1rem;display:flex;justify-content:center;
                  gap:0.6rem;flex-wrap:wrap">
        <a href="{PAPER_URL}" target="_blank"
           style="{btn};background:#0f172a;color:#ffffff">
           \U0001f4c4 Paper</a>
        <a href="{GITHUB_URL}" target="_blank"
           style="{btn};background:#f1f5f9;color:#334155">
           \U0001f4bb GitHub</a>
        <a href="{HF_URL}" target="_blank"
           style="{btn};background:#f1f5f9;color:#334155">
           \U0001f917 HuggingFace</a>
      </div>
      <div style="margin-top:1rem;display:flex;justify-content:center;
                  gap:1.5rem;flex-wrap:wrap">
        <span style="font-size:0.78rem;color:#94a3b8">
          76 tasks &middot; 5 molecular families</span>
        <span style="font-size:0.78rem;color:#94a3b8">
          17 MCP tools</span>
        <span style="font-size:0.78rem;color:#94a3b8">
          {n_entries} conditions</span>
        <span style="font-size:0.78rem;color:#94a3b8">
          Updated {last_updated}</span>
      </div>
    </div>"""


# ── Score styling helpers ──


def _score_color(s: float) -> str:
    if s >= 50:
        return "#38a169"
    if s >= 25:
        return "#d69e2e"
    return "#e53e3e"


def _bar_bg(s: float) -> str:
    if s >= 50:
        return "rgba(56,161,105,0.15)"
    if s >= 25:
        return "rgba(214,158,46,0.15)"
    return "rgba(229,62,62,0.12)"


def _heat_color(val, max_val=95) -> str:
    if val is None:
        return "#f7fafc"
    r = val / max_val
    if r >= 0.7:
        return f"rgba(56,161,105,{min(0.2 + r * 0.4, 0.8):.2f})"
    if r >= 0.4:
        return f"rgba(214,158,46,{min(0.2 + r * 0.4, 0.8):.2f})"
    return f"rgba(229,62,62,{min(0.15 + r * 0.3, 0.6):.2f})"


# ── Tab 1: Overall leaderboard table ──


def build_leaderboard_table(
    entries: list, mode_f: str, mcp_f: str, type_f: str
) -> str:
    """Generate the mixed-ranking HTML table with inline styles."""
    # Filter
    filtered = []
    for e in entries:
        st = e["submission_type"]
        if mode_f != "All" and st == "llm":
            if (e.get("mode") or "").lower() != mode_f.lower():
                continue
        if mcp_f == "Reference" and e.get("mcp_custom"):
            continue
        if mcp_f == "Custom" and not e.get("mcp_custom"):
            continue
        if type_f == "LLM Only" and st != "llm":
            continue
        if type_f == "Baselines Only" and st == "llm":
            continue
        filtered.append(e)

    filtered.sort(key=lambda x: x["overall_score"], reverse=True)

    # Shared cell styles
    TD = (
        "padding:0.65rem 1rem;border-bottom:1px solid #e2e8f0;"
        "font-size:0.9rem"
    )
    TH = (
        "background:#0f172a;color:white;padding:0.75rem 1rem;"
        "text-align:left;font-size:0.75rem;text-transform:uppercase;"
        "letter-spacing:0.05em;font-weight:600"
    )

    rows = []
    llm_rank = 0
    for e in filtered:
        st = e["submission_type"]
        sty = TYPE_STYLE.get(st, TYPE_STYLE["llm"])
        is_bl = st != "llm"
        sc = e["overall_score"]

        # ── Rank cell ──
        if is_bl:
            rank = (
                f'<td style="{TD};text-align:center;font-size:1.1rem;'
                f'width:50px">{sty["icon"]}</td>'
            )
        else:
            llm_rank += 1
            rcolor = {1: "#d69e2e", 2: "#a0aec0", 3: "#c17832"}.get(
                llm_rank, "#0f172a"
            )
            rsize = (
                "1.1rem"
                if llm_rank == 1
                else ("1.05rem" if llm_rank <= 3 else "0.9rem")
            )
            rank = (
                f'<td style="{TD};text-align:center;font-weight:700;'
                f"color:{rcolor};font-size:{rsize};width:50px\">"
                f"{llm_rank}</td>"
            )

        # ── Name cell ──
        tag_html = ""
        if sty["tag"]:
            tag_html = (
                ' <span style="font-size:0.7rem;background:#e2e8f0;'
                "padding:0.1rem 0.4rem;border-radius:3px;color:#4a5568;"
                f'margin-left:0.3rem;vertical-align:middle">'
                f'{sty["tag"]}</span>'
            )
        icon_pfx = f'{sty["icon"]} ' if sty["icon"] else ""
        fw = "600" if is_bl else "500"
        name = (
            f'<td style="{TD};font-weight:{fw}">'
            f'{icon_pfx}{e["agent_name"]}{tag_html}</td>'
        )

        # ── Organization ──
        org = f'<td style="{TD}">{e["organization"]}</td>'

        # ── Mode badge ──
        if is_bl:
            mode = f'<td style="{TD};color:#718096">\u2014</td>'
        elif e.get("mode") == "benchmark":
            mode = (
                f'<td style="{TD}"><span style="background:#fed7d7;'
                "color:#c53030;padding:0.15rem 0.5rem;border-radius:4px;"
                'font-size:0.75rem;font-weight:600">benchmark</span></td>'
            )
        else:
            mode = (
                f'<td style="{TD}"><span style="background:#c6f6d5;'
                "color:#276749;padding:0.15rem 0.5rem;border-radius:4px;"
                'font-size:0.75rem;font-weight:600">user</span></td>'
            )

        # ── MCP ──
        if is_bl:
            mcp = f'<td style="{TD};color:#718096">\u2014</td>'
        elif e.get("mcp_custom"):
            mcp = (
                f'<td style="{TD}"><span style="background:#fef3c7;'
                "color:#92400e;padding:0.15rem 0.55rem;border-radius:4px;"
                'font-size:0.72rem;font-weight:700">custom</span></td>'
            )
        else:
            mcp = (
                f'<td style="{TD}"><span style="background:#dbeafe;'
                "color:#1e40af;padding:0.15rem 0.55rem;border-radius:4px;"
                'font-size:0.72rem;font-weight:700">reference</span></td>'
            )

        # ── Score with proportional bar ──
        scol = _score_color(sc)
        bbg = _bar_bg(sc)
        score_cell = (
            f'<td style="{TD};font-weight:700;font-size:1rem;color:{scol};'
            f'position:relative;font-variant-numeric:tabular-nums">'
            f'<div style="position:absolute;left:0;top:0;bottom:0;'
            f"width:{sc}%;background:{bbg};"
            f'border-radius:3px"></div>'
            f'<span style="position:relative">{sc:.1f}</span></td>'
        )

        # ── Tasks & zeros ──
        tc = e.get("tasks_completed", 0)
        tt = e.get("tasks_total", 76)
        tasks = f'<td style="{TD}">{tc}/{tt}</td>'
        zeros = f'<td style="{TD}">{e.get("tasks_with_zero", 0)}</td>'

        rows.append(
            f'<tr style="background:{sty["bg"]}">'
            f"{rank}{name}{org}{mode}{mcp}{score_cell}{tasks}{zeros}</tr>"
        )

    return f"""
    <table style="width:100%;border-collapse:collapse;background:white;
                  border-radius:10px;overflow:hidden;
                  box-shadow:0 1px 3px rgba(0,0,0,0.08)">
      <thead><tr>
        <th style="{TH};width:50px">#</th>
        <th style="{TH}">Agent</th>
        <th style="{TH}">Organization</th>
        <th style="{TH}">Mode</th>
        <th style="{TH}">MCP</th>
        <th style="{TH}">Score</th>
        <th style="{TH}">Tasks</th>
        <th style="{TH}">Zero-Score</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


# ── Tab 2: Taxonomy heatmap ──


def build_heatmap(entry: dict) -> str:
    """HTML heatmap for one agent across the 2 × 5 design matrix
    (DesignApproach × MolecularSubject = 9 valid cells; rd × binder is empty).
    """
    ts = entry.get("taxonomy_scores", {})
    TH = (
        "background:#0f172a;color:white;padding:0.6rem 0.8rem;"
        "text-align:center;font-size:0.75rem;font-weight:600"
    )
    TD = (
        "text-align:center;padding:0.5rem;font-size:0.85rem;"
        "font-weight:600;border-bottom:1px solid #e2e8f0"
    )

    rows = []
    for ap in APPROACHES:
        cells = [
            f'<td style="{TD};text-align:left;font-weight:700;'
            f'background:#f8fafc;color:#0f172a">{APPROACH_LABELS[ap]}</td>'
        ]
        vals = []
        for sj in SUBJECTS:
            if sj in VALID_CELLS[ap]:
                val = ts.get(ap, {}).get(sj)
                bg = _heat_color(val)
                n = N_TASKS_PER_CELL.get((ap, sj), 0)
                text = (
                    f'{val:.0f}<br><span style="font-size:0.65rem;'
                    f'font-weight:400;color:#64748b">n={n}</span>'
                    if val is not None
                    else "\u2014"
                )
                cells.append(f'<td style="{TD};background:{bg}">{text}</td>')
                if val is not None:
                    vals.append(val)
            else:
                cells.append(
                    f'<td style="{TD};color:#cbd5e0;font-weight:400">'
                    "n/a</td>"
                )
        avg = sum(vals) / len(vals) if vals else 0
        avg_bg = _heat_color(avg)
        cells.append(
            f'<td style="{TD};font-weight:700;background:{avg_bg}">'
            f"{avg:.1f}</td>"
        )
        rows.append(f'<tr>{"".join(cells)}</tr>')

    sj_headers = "".join(
        f'<th style="{TH}">{SUBJECT_LABELS[sj]}</th>'
        for sj in SUBJECTS
    )

    return f"""
    <table style="width:100%;border-collapse:collapse;background:white;
                  border-radius:10px;overflow:hidden;
                  box-shadow:0 1px 3px rgba(0,0,0,0.08)">
      <thead><tr>
        <th style="{TH};text-align:left">Approach \u2193 / Subject \u2192</th>
        {sj_headers}
        <th style="{TH}">Mean</th>
      </tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>"""


# ── Tab 4: Mode comparison cards ──


def build_mode_cards(entries: list) -> str:
    """Per-LLM cards showing benchmark vs user delta."""
    by_name: dict[str, dict] = {}
    for e in entries:
        if e["submission_type"] != "llm":
            continue
        by_name.setdefault(e["agent_name"], {})[e["mode"]] = e

    ordered = sorted(
        by_name.items(),
        key=lambda x: x[1].get("user", {}).get("overall_score", 0),
        reverse=True,
    )

    cards = []
    for name, modes in ordered:
        bench = modes.get("benchmark")
        user = modes.get("user")
        if not bench or not user:
            continue
        delta = user["overall_score"] - bench["overall_score"]
        pct = (delta / bench["overall_score"] * 100) if bench["overall_score"] else 0

        lines = [
            '<div style="display:flex;justify-content:space-between;'
            'padding:0.4rem 0;border-bottom:1px solid #e2e8f0">'
            "<span>Benchmark</span>"
            f'<span style="font-weight:700;color:#e53e3e">'
            f'{bench["overall_score"]:.1f}</span></div>',
            '<div style="display:flex;justify-content:space-between;'
            'padding:0.4rem 0;border-bottom:1px solid #e2e8f0">'
            "<span>User</span>"
            f'<span style="font-weight:700;color:#d69e2e">'
            f'{user["overall_score"]:.1f}</span></div>',
            '<div style="display:flex;justify-content:space-between;'
            'padding:0.4rem 0;border-bottom:1px solid #e2e8f0">'
            "<span>Delta</span>"
            f'<span style="font-weight:700;color:#38a169">'
            f"+{delta:.1f} (+{pct:.0f}%)</span></div>",
        ]
        for c in COMPONENTS:
            d = user["component_scores"][c] - bench["component_scores"][c]
            color = "#38a169" if d >= 0 else "#e53e3e"
            sign = "+" if d >= 0 else ""
            lines.append(
                '<div style="display:flex;justify-content:space-between;'
                'padding:0.3rem 0;border-bottom:1px solid #e2e8f0;'
                'font-size:0.85rem">'
                f'<span style="color:#718096">{c}</span>'
                f'<span style="font-weight:700;color:{color}">'
                f"{sign}{d:.1f}</span></div>"
            )

        cards.append(
            '<div style="background:white;border-radius:10px;padding:1.2rem;'
            'box-shadow:0 1px 3px rgba(0,0,0,0.08)">'
            f'<h4 style="font-size:0.95rem;color:#0f172a;'
            f'margin:0 0 0.8rem">{name}</h4>'
            f'{"".join(lines)}</div>'
        )

    return (
        '<div style="display:grid;grid-template-columns:'
        'repeat(auto-fit,minmax(250px,1fr));gap:1rem;margin-top:1rem">'
        f'{"".join(cards)}</div>'
    )


# ── Headline findings (paper banner) ──


def build_headline_findings(findings: list) -> str:
    """Top-of-page banner that surfaces the paper's three core claims."""
    if not findings:
        return ""
    cards = []
    accents = ["#3182ce", "#d69e2e", "#805ad5", "#38a169", "#e53e3e"]
    for i, text in enumerate(findings):
        c = accents[i % len(accents)]
        cards.append(
            f'<div style="background:#ffffff;border:1px solid #e2e8f0;'
            f"border-left:4px solid {c};border-radius:10px;"
            f'padding:0.85rem 1rem;flex:1 1 220px;min-width:220px;'
            f'box-shadow:0 1px 3px rgba(0,0,0,0.04)">'
            f'<div style="font-size:0.7rem;font-weight:700;'
            f'color:{c};letter-spacing:0.08em;text-transform:uppercase;'
            f'margin-bottom:0.35rem">Finding {i+1}</div>'
            f'<div style="font-size:0.82rem;color:#1a202c;'
            f'line-height:1.45">{text}</div></div>'
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:0.7rem;'
        'margin:0.4rem 0 1rem">'
        f"{''.join(cards)}</div>"
    )


# ── Tab: Depth Gap (intervention experiments) ──


def build_intervention_section(interventions: dict) -> str:
    """Show forced-depth and low-diversity intervention results.

    The forced-depth condition mandates ≥3 evaluation passes per design
    candidate; the low-diversity control constrains the candidate pool
    without forcing depth. Together they isolate evaluation depth as the
    causal driver of the 'surface competence' gap reported in the paper.
    """
    if not interventions or not interventions.get("rows"):
        return '<p style="color:#718096">No intervention data available.</p>'

    rows = interventions["rows"]

    cond_meta = {
        "baseline": ("#64748b", "Baseline"),
        "forced_depth": ("#38a169", "Forced Depth"),
        "low_diversity_control": ("#d69e2e", "Low-Diversity Control"),
    }

    TH = (
        "background:#0f172a;color:white;padding:0.65rem 0.9rem;"
        "text-align:left;font-size:0.72rem;text-transform:uppercase;"
        "letter-spacing:0.05em;font-weight:600"
    )
    TD = ("padding:0.6rem 0.9rem;border-bottom:1px solid #e2e8f0;"
          "font-size:0.86rem")

    body = []
    for r in rows:
        color, cond_label = cond_meta.get(r["condition"], ("#64748b", r["condition"]))
        delta = r.get("delta_vs_baseline")
        if delta is None or r["condition"] == "baseline":
            delta_html = '<span style="color:#cbd5e0">\u2014</span>'
        else:
            sign = "+" if delta >= 0 else ""
            dcol = "#38a169" if delta > 0 else ("#e53e3e" if delta < 0 else "#64748b")
            delta_html = (
                f'<span style="color:{dcol};font-weight:700">'
                f"{sign}{delta:.1f}</span>"
            )
        body.append(
            f'<tr><td style="{TD};font-weight:600;color:#0f172a">'
            f'{r["label"]}</td>'
            f'<td style="{TD}"><span style="background:{color}22;'
            f"color:{color};padding:0.15rem 0.55rem;border-radius:4px;"
            f'font-size:0.72rem;font-weight:700">{cond_label}</span></td>'
            f'<td style="{TD};font-weight:700;font-variant-numeric:'
            f'tabular-nums">{r["score"]:.1f}</td>'
            f'<td style="{TD};font-variant-numeric:tabular-nums">{delta_html}</td>'
            f'<td style="{TD};color:#475569;font-variant-numeric:tabular-nums">'
            f'{r["approach"]:.1f} / {r["orchestration"]:.1f}</td>'
            f'<td style="{TD};color:#475569;font-variant-numeric:tabular-nums">'
            f'{r["quality"]:.1f}</td>'
            f'<td style="{TD};color:#475569;font-variant-numeric:tabular-nums">'
            f'{r["diversity"]:.1f}</td></tr>'
        )

    n = interventions.get("n_tasks", 18)

    return f"""
    <div style="max-width:980px;margin:0 auto">

      <div style="background:#ffffff;border:1px solid #e2e8f0;
                  border-radius:12px;padding:1.4rem 1.6rem;
                  margin-bottom:1rem">
        <h2 style="color:#0f172a;margin:0 0 0.5rem;font-size:1.2rem;
                   font-weight:700">Causal interventions on the depth gap</h2>
        <p style="color:#475569;line-height:1.55;margin:0">
          {interventions.get('description', '')}
          Reruns are scored on a representative <strong>{n}-task</strong>
          subset that spans all 9 occupied taxonomy cells.
        </p>
      </div>

      <div style="background:#fefce8;border-left:4px solid #ca8a04;
                  border-radius:8px;padding:0.95rem 1.1rem;
                  margin-bottom:1.1rem">
        <strong style="color:#713f12">Headline:</strong>
        <span style="color:#52340d">
          Forced-depth lifts <strong>DeepSeek V3 by +9.3</strong> and
          <strong>GPT-5 by +15.9</strong> points without any change to
          the underlying model or tools, while the low-diversity control
          <em>hurts</em> DeepSeek V3 (&minus;2.3). The dissociation is
          cleanest on the strongest agent, where it provides direct
          causal evidence that
          <strong>evaluation depth &mdash; not the mere act of process
          intervention &mdash; drives the gain</strong>. GPT-5's
          response is more uniform across both interventions; we
          report the raw deltas without smoothing.
        </span>
      </div>

      <table style="width:100%;border-collapse:collapse;background:white;
                    border-radius:10px;overflow:hidden;
                    box-shadow:0 1px 3px rgba(0,0,0,0.08)">
        <thead><tr>
          <th style="{TH}">Run</th>
          <th style="{TH}">Condition</th>
          <th style="{TH}">Score</th>
          <th style="{TH}">&Delta; vs baseline</th>
          <th style="{TH}">Approach / Orch.</th>
          <th style="{TH}">Quality</th>
          <th style="{TH}">Diversity</th>
        </tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>

      <p style="color:#64748b;font-size:0.78rem;margin-top:0.8rem;
                line-height:1.5">
        Scoring uses the same 100-point hybrid rubric as the main
        leaderboard but is restricted to {n} representative tasks;
        absolute values therefore differ from the full-benchmark mean.
        The <em>delta vs baseline</em> compares each agent against
        its own untreated baseline run, isolating the intervention effect.
      </p>
    </div>
    """


# ── Tab 5: About ──


def build_about() -> str:
    h2 = (
        'style="color:#0f172a;margin:0 0 0.8rem;font-size:1.25rem;'
        'font-weight:700"'
    )
    h3 = (
        'style="color:#334155;margin:1.2rem 0 0.5rem;font-size:1rem;'
        'font-weight:600"'
    )
    p = 'style="margin-bottom:0.8rem;color:#475569;line-height:1.6"'
    card = (
        'style="background:#ffffff;border:1px solid #e2e8f0;'
        'border-radius:12px;padding:2rem;margin-bottom:1.2rem"'
    )
    stat_box = (
        'style="background:#f8fafc;border:1px solid #e2e8f0;'
        'border-radius:10px;padding:1rem;text-align:center"'
    )
    return f"""
    <div style="max-width:900px;margin:0 auto">

      <div {card}>
        <h2 {h2}>What is BioDesignBench?</h2>
        <p {p}>
          BioDesignBench is a benchmark for evaluating LLM agents as
          orchestrators of multi-step <em>stochastic</em> protein-design
          pipelines. Unlike chemistry- or code-agent benchmarks, where
          tool chains are largely deterministic, protein design demands
          repeated sampling from generative tools (RFdiffusion,
          ProteinMPNN) and iterative cross-validation through several
          biophysical metrics. We test the full agentic loop &mdash;
          <strong>plan &rarr; sample &rarr; evaluate across multiple
          metrics &rarr; iterate</strong> &mdash; over 76 expert-curated
          tasks drawn from 2024&ndash;2026 literature, exposed through
          17 MCP-integrated tools.
        </p>
        <div style="display:grid;grid-template-columns:
                    repeat(auto-fit,minmax(140px,1fr));gap:0.8rem;
                    margin:1rem 0">
          <div {stat_box}>
            <div style="font-size:1.8rem;font-weight:800;color:#0f172a">
              76</div>
            <div style="font-size:0.78rem;color:#64748b">design tasks</div>
          </div>
          <div {stat_box}>
            <div style="font-size:1.8rem;font-weight:800;color:#0f172a">
              9</div>
            <div style="font-size:0.78rem;color:#64748b">
              taxonomy cells<br>(2 approaches \u00d7 5 subjects)</div>
          </div>
          <div {stat_box}>
            <div style="font-size:1.8rem;font-weight:800;color:#0f172a">
              17</div>
            <div style="font-size:0.78rem;color:#64748b">MCP tools</div>
          </div>
          <div {stat_box}>
            <div style="font-size:1.8rem;font-weight:800;color:#0f172a">
              100</div>
            <div style="font-size:0.78rem;color:#64748b">point rubric</div>
          </div>
        </div>
      </div>

      <div {card}>
        <h2 {h2}>Three principal findings</h2>
        <h3 {h3}>1. Top-tier agents now beat a deterministic pipeline</h3>
        <p {p}>
          DeepSeek V3 and GPT-5 surpass a hand-engineered hardcoded
          pipeline (54.2) under both modes. Autonomous protein-design
          orchestration is no longer infeasible &mdash; but a substantial
          gap to the human expert (61.3) and oracle (74.9) remains.
        </p>
        <h3 {h3}>2. Coverage&ndash;depth dissociation</h3>
        <p {p}>
          Workflow guidance closes the <em>coverage</em> gap (Rescue
          Index up to +3.01) but leaves <em>utilisation depth</em>
          unchanged (Rescue Index \u2248 0). Better tool documentation
          can teach agents <em>which</em> tools to call, but cannot
          teach them to call those tools with the iterative depth that
          expert practice demands.
        </p>
        <h3 {h3}>3. Evaluation depth, not tool knowledge, is the bottleneck</h3>
        <p {p}>
          Across 836 task&ndash;condition observations, evaluation depth
          per candidate correlates with total score at
          <strong>&rho; = 0.685</strong>
          (<em>p</em> &lt; 10<sup>-117</sup>). LLM agents generate
          backbone candidates at expert-level rates but evaluate each
          one at only <strong>14% of expert depth</strong>. Forced-depth
          interventions confirm this is causal &mdash; see the
          <em>Depth Gap</em> tab.
        </p>
      </div>

      <div {card}>
        <h2 {h2}>How to submit</h2>
        <h3 {h3}>1. Build your agent</h3>
        <p {p}>
          Create a protein design agent that runs the full plan &rarr;
          sample &rarr; evaluate &rarr; iterate loop on each task. Pick one
          of two MCP options:</p>
        <ul style="color:#475569;padding-left:1.5rem;margin-bottom:0.8rem;
                   line-height:1.7">
          <li><strong>Reference MCP</strong> &mdash; connect to our published
            <a href="https://github.com/RomeroLab/protein-design-mcp"
               style="color:#2563eb;font-weight:600">protein-design-mcp</a>
            server (Docker image / Modal endpoint, in progress). Eligible for
            the reference ranking.</li>
          <li><strong>Custom MCP</strong> &mdash; bring your own tool
            implementations. Tagged with a <code>custom</code> badge on the
            leaderboard, excluded from the reference ranking.</li>
        </ul>
        <h3 {h3}>2. Host an API endpoint</h3>
        <p {p}>
          Your agent must be accessible as a POST endpoint that accepts
          task payloads and returns designed sequences plus a tool-call
          trace. See <code>biodesignbench-leaderboard/example_server.py</code>
          for a 200-line reference.</p>
        <h3 {h3}>API specification</h3>
        <pre style="background:#0f172a;color:#e2e8f0;padding:1.2rem;
                    border-radius:10px;font-size:0.8rem;overflow-x:auto;
                    line-height:1.6">POST /api/run

Request:
{{
  "task_id": "dnb_ab_001",
  "task_description": "Design a de novo binder for...",
  "available_tools": [...],
  "input_files": {{ "<pdb-name>": "<base64>" }},
  "design_constraints": {{ ... }},
  "max_steps": 50,
  "timeout_sec": 300
}}

Response:
{{
  "sequences": ["MKKL..."],
  "run_log": [{{ "step": 1, "tool": "...", "success": true }}],
  "total_steps": 12,
  "total_time_sec": 142.5,
  "metrics": {{}}
}}</pre>
        <h3 {h3}>3. Submit and wait</h3>
        <p {p}>
          We dispatch 73 hidden tasks to your endpoint, run Boltz-2
          structure verification on each design, and score against the
          100-point hybrid rubric (algorithmic + 3-judge LLM panel).
          Maximum <strong>1 submission per month</strong> per
          organization &mdash; LLM-judge API costs are paid by Romero
          Lab.</p>
        <p {p}>
          3 example tasks are publicly available for development and
          testing your endpoint before submission.</p>

        <h3 {h3}>Limits</h3>
        <ul style="color:#475569;padding-left:1.5rem;margin-bottom:0.8rem;
                   line-height:1.7">
          <li>Maximum 1 submission per calendar month per organization</li>
          <li>73 hidden tasks are used for ranking</li>
          <li>3 public example tasks are available for development</li>
        </ul>
      </div>

      <div {card}>
        <h2 {h2}>Scoring rubric (100 points, hybrid)</h2>
        <p {p}>
          Scores combine <strong>72 algorithmic points</strong> from
          deterministic biophysical metrics with
          <strong>28 LLM-judge points</strong> assessed by a 3-judge
          panel (PoLL) with self-exclusion to mitigate self-preference
          bias. Each component is capped at its rubric maximum to
          prevent double counting.
        </p>
        <p {p}>
          <strong>Approach (20 pts)</strong> &mdash; strategic
          appropriateness of tool selection across 10 functional
          categories (backbone generation, inverse folding, structure
          prediction, etc.).</p>
        <p {p}>
          <strong>Orchestration (15 pts)</strong> &mdash; pipeline
          ordering, intermediate validation, and adaptive iteration.</p>
        <p {p}>
          <strong>Quality (35 pts)</strong> &mdash; 100% algorithmic.
          Continuous 4-band interpolation over Boltz-2 re-prediction
          metrics (pLDDT, pTM, ipTM, i_pAE), eliminating LLM judgement
          variance on biophysical quantities.</p>
        <p {p}>
          <strong>Feasibility (15 pts)</strong> &mdash; valid amino
          acids, length constraints, composition, and biophysical
          plausibility.</p>
        <p {p}>
          <strong>Novelty (5 pts)</strong> &mdash; sequence identity to
          reference (lower identity = more novel).</p>
        <p {p}>
          <strong>Diversity (10 pts)</strong> &mdash; number and
          pairwise diversity of generated designs.</p>
      </div>

      <div {card}>
        <h2 {h2}>Five-layer contamination defense</h2>
        <p {p}>Every evaluated LLM may have read protein-design
          literature during pretraining, so we use a layered defense:</p>
        <ul style="color:#475569;padding-left:1.5rem;
                   margin-bottom:0.8rem;line-height:1.7">
          <li>All 76 tasks derived from publications dated 2024&ndash;2026,
              post-dating model training cutoffs.</li>
          <li>Task prompts paraphrased and restructured &mdash; no
              verbatim passages from source literature.</li>
          <li>Targets specified by biological function and structural
              constraints, not by name or PDB identifier.</li>
          <li>12 decoy tasks with deliberately fabricated targets to
              detect memorisation-based responses.</li>
          <li>n-gram overlap analysis between agent outputs and source
              publications &mdash; no verbatim regurgitation above the
              8-gram threshold across any condition.</li>
        </ul>
      </div>

      <div {card}>
        <h2 {h2}>Citation</h2>
        <pre style="background:#0f172a;color:#e2e8f0;padding:1.2rem;
                    border-radius:10px;font-size:0.8rem;
                    line-height:1.6">@article{{biodesignbench2026,
  title={{Evaluating LLM-Driven Protein Design:
         Agents Lack Iterative Evaluation Depth}},
  author={{Kim, Jeonghyeon and Romero, Philip}},
  year={{2026}}
}}</pre>
      </div>

    </div>"""


# ═══════════════════════════════════════════════════════════════════
#  Chart builders (Plotly)
# ═══════════════════════════════════════════════════════════════════


def chart_taxonomy_bar(entry: dict) -> go.Figure:
    """Grouped bar chart of mean score per molecular subject,
    split by design approach (de novo vs redesign).
    """
    ts = entry.get("taxonomy_scores", {})
    x_labels = [SUBJECT_LABELS[s] for s in SUBJECTS]

    def _series(ap):
        out = []
        for sj in SUBJECTS:
            if sj in VALID_CELLS[ap]:
                out.append(ts.get(ap, {}).get(sj))
            else:
                out.append(None)
        return out

    dn = _series("de_novo")
    rd = _series("redesign")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x_labels, y=dn, name="De Novo",
        marker_color="rgba(49,130,206,0.78)",
        text=[f"{v:.0f}" if v is not None else "" for v in dn],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=x_labels, y=rd, name="Redesign",
        marker_color="rgba(214,158,46,0.78)",
        text=[f"{v:.0f}" if v is not None else "" for v in rd],
        textposition="outside",
    ))
    mode = entry.get("mode") or "\u2014"
    fig.update_layout(
        **_base_layout(
            barmode="group",
            title=dict(
                text=f"{entry['agent_name']} ({mode}) \u2014 Mean Score by Cell",
                font_size=14,
            ),
            yaxis=dict(range=[0, 100], title="Hybrid score (out of 100)"),
            xaxis=dict(title=""),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2,
                        xanchor="center", x=0.5),
            height=340,
        )
    )
    return fig


def chart_radar(e1: dict, e2: dict) -> go.Figure:
    """Radar chart comparing two agents' component scores (% of max)."""
    labels = [c.capitalize() for c in COMPONENTS]

    def norm(e):
        return [e["component_scores"][c] / COMP_MAX[c] * 100 for c in COMPONENTS]

    v1, v2 = norm(e1), norm(e2)
    m1 = e1.get("mode") or "\u2014"
    m2 = e2.get("mode") or "\u2014"

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=v1 + [v1[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=f'{e1["agent_name"]} ({m1})',
            line=dict(color="rgba(49,130,206,0.8)"),
            fillcolor="rgba(49,130,206,0.15)",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=v2 + [v2[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=f'{e2["agent_name"]} ({m2})',
            line=dict(color="rgba(229,62,62,0.8)"),
            fillcolor="rgba(229,62,62,0.15)",
        )
    )
    fig.update_layout(
        **_base_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], ticksuffix="%")
            ),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.25,
                xanchor="center", x=0.5,
            ),
            title=dict(text="Component Radar (% of max)", font_size=14),
            height=420,
        )
    )
    return fig


def chart_component_bar(e1: dict, e2: dict) -> go.Figure:
    """Horizontal bar chart of raw component scores for two agents."""
    labels = [f"{c.capitalize()} (/{COMP_MAX[c]})" for c in COMPONENTS]
    m1 = e1.get("mode") or "\u2014"
    m2 = e2.get("mode") or "\u2014"

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[e1["component_scores"][c] for c in COMPONENTS],
            name=f'{e1["agent_name"]} ({m1})',
            orientation="h",
            marker_color="rgba(49,130,206,0.7)",
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=[e2["component_scores"][c] for c in COMPONENTS],
            name=f'{e2["agent_name"]} ({m2})',
            orientation="h",
            marker_color="rgba(229,62,62,0.7)",
        )
    )
    fig.update_layout(
        **_base_layout(
            barmode="group",
            xaxis=dict(title="Score"),
            title=dict(text="Component Breakdown", font_size=14),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.3,
                xanchor="center", x=0.5,
            ),
            height=420,
        )
    )
    return fig


def chart_mode_comparison(entries: list) -> go.Figure:
    """Grouped bar chart: benchmark vs user mode for each LLM."""
    by_name: dict[str, dict[str, float]] = {}
    for e in entries:
        if e["submission_type"] != "llm":
            continue
        by_name.setdefault(e["agent_name"], {})[e["mode"]] = e["overall_score"]

    ordered = sorted(
        by_name.items(),
        key=lambda x: x[1].get("user", 0),
        reverse=True,
    )
    names = [n for n, _ in ordered]
    bench = [m.get("benchmark", 0) for _, m in ordered]
    user = [m.get("user", 0) for _, m in ordered]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=names, y=bench, name="Benchmark Mode",
            marker_color="rgba(229,62,62,0.6)",
        )
    )
    fig.add_trace(
        go.Bar(
            x=names, y=user, name="User Mode",
            marker_color="rgba(56,161,105,0.6)",
        )
    )
    fig.update_layout(
        **_base_layout(
            barmode="group",
            yaxis=dict(range=[0, 80], title="Overall hybrid score"),
            xaxis=dict(title=""),
            title=dict(
                text=("Unguided (Benchmark) vs Guided (User) modes \u2014 "
                      "guidance lifts coverage but rarely shifts overall score"),
                font_size=13,
            ),
            legend=dict(
                orientation="h", yanchor="bottom", y=-0.18,
                xanchor="center", x=0.5,
            ),
            height=380,
        )
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
#  Gradio application
# ═══════════════════════════════════════════════════════════════════


def create_app() -> gr.Blocks:
    data = load_data()
    entries = data["entries"]
    by_id = {e["agent_id"]: e for e in entries}

    # Build dropdown choices: (display_label, agent_id)
    agent_choices = []
    for e in entries:
        sty = TYPE_STYLE.get(e["submission_type"], TYPE_STYLE["llm"])
        icon = sty["icon"]
        mode = e.get("mode") or "\u2014"
        label = f"{icon} {e['agent_name']} ({mode})".strip()
        agent_choices.append((label, e["agent_id"]))

    # Safe index helper
    def _choice_val(idx: int) -> str:
        return agent_choices[min(idx, len(agent_choices) - 1)][1]

    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue"),
        css=CUSTOM_CSS,
        js=FORCE_LIGHT_JS,
    ) as app:

        gr.HTML(build_header(data["last_updated"], len(entries)))
        gr.HTML(build_headline_findings(data.get("headline_findings", [])))

        with gr.Tabs():

            # ════════ Tab 1: Overall Leaderboard ════════
            with gr.Tab("\U0001f4ca Overall"):
                with gr.Row():
                    f_mode = gr.Dropdown(
                        ["All", "Benchmark", "User"],
                        value="All", label="Mode", scale=1,
                    )
                    f_mcp = gr.Dropdown(
                        ["All", "Reference", "Custom"],
                        value="All", label="MCP Tools", scale=1,
                    )
                    f_type = gr.Dropdown(
                        ["All Entries", "LLM Only", "Baselines Only"],
                        value="All Entries", label="Show", scale=1,
                    )

                tbl = gr.HTML(
                    build_leaderboard_table(
                        entries, "All", "All", "All Entries"
                    )
                )

                def _update_table(m, mc, t):
                    return build_leaderboard_table(entries, m, mc, t)

                for dd in [f_mode, f_mcp, f_type]:
                    dd.change(
                        _update_table, [f_mode, f_mcp, f_type], tbl
                    )

            # ════════ Tab 2: Taxonomy Breakdown ════════
            with gr.Tab("\U0001f9ec Taxonomy"):
                tax_dd = gr.Dropdown(
                    agent_choices,
                    value=_choice_val(0),
                    label="Select Agent",
                )
                hm_html = gr.HTML(build_heatmap(entries[0]))
                tax_plot = gr.Plot(chart_taxonomy_bar(entries[0]))

                def _update_taxonomy(aid):
                    e = by_id.get(aid, entries[0])
                    return build_heatmap(e), chart_taxonomy_bar(e)

                tax_dd.change(
                    _update_taxonomy, [tax_dd], [hm_html, tax_plot]
                )

            # ════════ Tab 3: Component Analysis ════════
            with gr.Tab("\U0001f3af Components"):
                with gr.Row():
                    c1 = gr.Dropdown(
                        agent_choices, value=_choice_val(0),
                        label="Agent 1", scale=1,
                    )
                    c2 = gr.Dropdown(
                        agent_choices, value=_choice_val(4),
                        label="Agent 2", scale=1,
                    )
                with gr.Row():
                    radar = gr.Plot(
                        chart_radar(
                            entries[0],
                            entries[min(4, len(entries) - 1)],
                        )
                    )
                    comp_bar = gr.Plot(
                        chart_component_bar(
                            entries[0],
                            entries[min(4, len(entries) - 1)],
                        )
                    )

                def _update_comp(a1, a2):
                    e1 = by_id.get(a1, entries[0])
                    e2 = by_id.get(a2, entries[-1])
                    return chart_radar(e1, e2), chart_component_bar(e1, e2)

                for dd in [c1, c2]:
                    dd.change(_update_comp, [c1, c2], [radar, comp_bar])

            # ════════ Tab 4: Benchmark vs User (coverage-depth dissociation) ════════
            with gr.Tab("\u26a1 Guidance Effect"):
                gr.HTML(
                    '<div style="background:#eff6ff;border-left:4px solid '
                    '#3182ce;border-radius:8px;padding:0.85rem 1.1rem;'
                    'margin:0.4rem 0 0.9rem;color:#1e3a8a;font-size:0.88rem;'
                    'line-height:1.55">'
                    '<strong>Mode semantics:</strong> '
                    '<em>Benchmark mode</em> exposes atomic tools without '
                    'pipeline hints (unguided); <em>User mode</em> packages '
                    'them into composite workflows with explicit pipeline '
                    'structure (guided). Guidance lifts the lowest-tier '
                    'agents but does not consistently help capable ones, '
                    'and never closes the depth gap (see <em>Depth Gap</em> '
                    'tab).</div>'
                )
                gr.Plot(chart_mode_comparison(entries))
                gr.HTML(build_mode_cards(entries))

            # ════════ Tab 5: Depth Gap (interventions) ════════
            with gr.Tab("\U0001f50d Depth Gap"):
                gr.HTML(build_intervention_section(
                    data.get("interventions", {})
                ))

            # ══════ Tab 5: Submit ══════
            with gr.Tab("\U0001f4e4 Submit"):
                gr.HTML("""
                <div style="max-width:780px;margin:0 auto;padding:1rem">
                  <h2 style="color:#0f172a;margin:0 0 0.5rem;
                             font-weight:700;font-size:1.25rem">
                    Submit your agent</h2>
                  <p style="color:#475569;margin-bottom:1rem;line-height:1.6">
                    Host your protein-design agent as an HTTPS endpoint that
                    accepts task payloads and returns designed sequences plus
                    a tool-call trace. The leaderboard will POST each of the
                    76 hidden tasks to your endpoint, run Boltz-2 structure
                    verification, score the rubric, and publish the result.
                  </p>

                  <div style="background:#eff6ff;border-left:4px solid #3182ce;
                              padding:0.95rem 1.1rem;border-radius:8px;
                              margin-bottom:1rem;font-size:0.86rem;
                              color:#1e3a8a;line-height:1.55">
                    <strong>Two MCP options &mdash; pick one below:</strong>
                    <ul style="margin:0.5rem 0 0 1.1rem;padding:0">
                      <li><strong>Reference MCP</strong> (recommended):
                        connect your agent to our published
                        <a href="https://github.com/RomeroLab/protein-design-mcp"
                           style="color:#1d4ed8;font-weight:600">protein-design-mcp</a>
                        Docker image / Modal endpoint so every submission uses
                        the identical 17-tool reference implementation.
                        Eligible for the <em>reference</em> ranking.
                      </li>
                      <li><strong>Custom MCP</strong>: bring your own tool
                        implementations. Tagged with a <code>custom</code>
                        badge and excluded from the reference ranking. Useful
                        for measuring tool-implementation contributions.
                      </li>
                    </ul>
                  </div>

                  <div style="background:#fefce8;border-left:3px solid #ca8a04;
                              padding:0.8rem 1rem;border-radius:6px;
                              margin-bottom:1rem;font-size:0.85rem;color:#713f12">
                    <strong>Rate limit:</strong> 1 submission per calendar
                    month per organization. LLM-judge API costs (~$10/run)
                    are paid by Romero Lab, so please be considerate.
                    You bear your own agent / tool compute costs.
                  </div>

                  <p style="color:#475569;font-size:0.85rem;line-height:1.55;
                            margin:0">
                    See
                    <code>biodesignbench-leaderboard/example_server.py</code>
                    in the
                    <a href="https://github.com/RomeroLab/BioDesignBench"
                       style="color:#2563eb;font-weight:500">GitHub repo</a>
                    for a 200-line reference implementation of the endpoint.
                  </p>
                </div>""")

                with gr.Column(scale=1):
                    sub_agent = gr.Textbox(
                        label="Agent Name",
                        placeholder="e.g., GPT-5 + protein-design-mcp",
                    )
                    sub_org = gr.Textbox(
                        label="Organization",
                        placeholder="e.g., OpenAI",
                    )
                    sub_url = gr.Textbox(
                        label="Endpoint URL",
                        placeholder="https://your-server.com/api/run",
                    )
                    sub_desc = gr.Textbox(
                        label="Description (optional)",
                        placeholder="Brief description of your agent...",
                        lines=3,
                    )
                    sub_mcp_mode = gr.Radio(
                        choices=[
                            ("Reference MCP (eligible for ranking)", "reference"),
                            ("Custom MCP (own tool implementations)", "custom"),
                        ],
                        value="reference",
                        label="MCP tool implementation",
                        info=(
                            "Reference = your agent calls our published "
                            "protein-design-mcp server. Custom = your agent "
                            "uses its own tool implementations."
                        ),
                    )
                    sub_btn = gr.Button(
                        "Submit for Review",
                        variant="primary",
                    )
                    sub_result = gr.HTML()

                def _handle_submit(name, org, url, desc, mcp_mode):
                    if not name or not org or not url:
                        return ('<div style="color:#e53e3e;padding:0.5rem">'
                                "Please fill in all required fields.</div>")
                    if not url.startswith(("http://", "https://")):
                        return ('<div style="color:#e53e3e;padding:0.5rem">'
                                "URL must start with http:// or https://</div>")
                    try:
                        from eval_queue import submit
                        result = submit(
                            agent_name=name,
                            organization=org,
                            endpoint_url=url,
                            description=desc,
                            mcp_custom=(mcp_mode == "custom"),
                        )
                        if "error" in result:
                            return (f'<div style="color:#e53e3e;padding:0.5rem">'
                                    f'{result["error"]}</div>')
                        return (
                            f'<div style="background:#c6f6d5;padding:1rem;'
                            f'border-radius:8px;margin-top:0.5rem">'
                            f'<strong>Submitted!</strong> '
                            f'ID: <code>{result["submission_id"]}</code><br>'
                            f'Status: {result["status"]}<br>'
                            f'MCP mode: <strong>{mcp_mode}</strong><br>'
                            f'{result.get("message", "")}</div>'
                        )
                    except Exception as e:
                        return (f'<div style="color:#e53e3e;padding:0.5rem">'
                                f"Error: {str(e)[:200]}</div>")

                sub_btn.click(
                    _handle_submit,
                    [sub_agent, sub_org, sub_url, sub_desc, sub_mcp_mode],
                    sub_result,
                )

            # ══════ Tab 6: Status & Admin ══════
            with gr.Tab("\U0001f6e0 Status"):
                gr.HTML("""
                <div style="max-width:800px;margin:0 auto;padding:1rem">
                  <h2 style="color:#0f172a;margin:0 0 0.5rem;
                             font-weight:700;font-size:1.25rem">
                    Submission status</h2>
                  <p style="color:#475569;margin-bottom:0.5rem;line-height:1.6">
                    Check your submission status or manage the pipeline
                    (admin only).</p>
                </div>""")

                # --- Public status check ---
                with gr.Accordion("Check Submission Status", open=True):
                    status_id = gr.Textbox(
                        label="Submission ID",
                        placeholder="Enter your submission ID...",
                    )
                    status_btn = gr.Button("Check Status")
                    status_out = gr.HTML()

                    def _check_status(sid):
                        if not sid:
                            return '<div style="color:#718096">Enter an ID above.</div>'
                        try:
                            from eval_queue import get_submission
                            sub = get_submission(sid.strip())
                            if sub is None:
                                return ('<div style="color:#e53e3e">'
                                        "Submission not found.</div>")
                            status_color = {
                                "pending": "#d69e2e", "approved": "#38a169",
                                "dispatching": "#3182ce", "boltz": "#805ad5",
                                "scoring": "#805ad5", "complete": "#38a169",
                                "failed": "#e53e3e", "rejected": "#e53e3e",
                            }.get(sub["status"], "#718096")
                            score_html = ""
                            if sub.get("overall_score") is not None:
                                score_html = (
                                    f'<div style="font-size:1.2rem;'
                                    f'font-weight:700;color:#0f172a;'
                                    f'margin-top:0.5rem">'
                                    f'Score: {sub["overall_score"]:.1f}/100'
                                    f'</div>'
                                )
                            return (
                                f'<div style="background:white;padding:1rem;'
                                f'border-radius:8px;border:1px solid #e2e8f0">'
                                f'<strong>{sub["agent_name"]}</strong> '
                                f'({sub["organization"]})<br>'
                                f'Status: <span style="color:{status_color};'
                                f'font-weight:700">{sub["status"]}</span><br>'
                                f'Tasks: {sub.get("tasks_dispatched", 0)}'
                                f'/{sub.get("tasks_total", 76)}<br>'
                                f'Created: {sub.get("created_at", "")[:10]}'
                                f'{score_html}</div>'
                            )
                        except Exception as e:
                            return f'<div style="color:#e53e3e">{e}</div>'

                    status_btn.click(_check_status, [status_id], status_out)

                # --- Admin panel (password-protected) ---
                with gr.Accordion("Admin Panel", open=False):
                    admin_pw = gr.Textbox(
                        label="Admin Password", type="password",
                    )
                    admin_auth_btn = gr.Button("Authenticate")
                    admin_panel = gr.Column(visible=False)
                    admin_msg = gr.HTML()

                    with admin_panel:
                        gr.HTML('<h3 style="color:#0f172a">'
                                'Pending Submissions</h3>')
                        pending_html = gr.HTML()
                        refresh_btn = gr.Button("Refresh List")

                        with gr.Row():
                            approve_id = gr.Textbox(
                                label="Submission ID to Approve/Reject",
                                scale=2,
                            )
                            approve_btn = gr.Button(
                                "Approve", variant="primary", scale=1,
                            )
                            reject_btn = gr.Button(
                                "Reject", variant="stop", scale=1,
                            )
                        approve_msg = gr.HTML()

                        gr.HTML('<h3 style="color:#0f172a;margin-top:1rem">'
                                'Pipeline Control</h3>')
                        with gr.Row():
                            dispatch_id = gr.Textbox(
                                label="Submission ID", scale=2,
                            )
                            dispatch_btn = gr.Button(
                                "Phase A: Dispatch Tasks", scale=1,
                            )
                        with gr.Row():
                            boltz_id = gr.Textbox(
                                label="Submission ID", scale=2,
                            )
                            boltz_btn = gr.Button(
                                "Phase B: Run Boltz (GPU)", scale=1,
                            )
                        with gr.Row():
                            judge_id = gr.Textbox(
                                label="Submission ID", scale=2,
                            )
                            judge_btn = gr.Button(
                                "Phase C: Run LLM Judge", scale=1,
                            )
                        with gr.Row():
                            final_id = gr.Textbox(
                                label="Submission ID", scale=2,
                            )
                            final_btn = gr.Button(
                                "Phase D: Finalize & Publish", scale=1,
                            )
                        pipeline_out = gr.HTML()

                    def _admin_auth(pw):
                        if pw == ADMIN_PASSWORD:
                            return (
                                gr.Column(visible=True),
                                '<div style="color:#38a169">'
                                'Authenticated.</div>',
                            )
                        return (
                            gr.Column(visible=False),
                            '<div style="color:#e53e3e">'
                            'Wrong password.</div>',
                        )

                    admin_auth_btn.click(
                        _admin_auth, [admin_pw],
                        [admin_panel, admin_msg],
                    )

                    def _refresh_pending():
                        try:
                            from eval_queue import get_pending_submissions
                            pending = get_pending_submissions()
                            if not pending:
                                return "<p>No pending submissions.</p>"
                            rows = []
                            for s in pending:
                                rows.append(
                                    f'<tr><td>{s["submission_id"]}</td>'
                                    f'<td>{s["agent_name"]}</td>'
                                    f'<td>{s["organization"]}</td>'
                                    f'<td>{s.get("endpoint_url","")[:40]}'
                                    f'...</td>'
                                    f'<td>{s.get("created_at","")[:10]}'
                                    f'</td></tr>'
                                )
                            return (
                                '<table style="width:100%;font-size:0.85rem;'
                                'border-collapse:collapse">'
                                "<tr><th>ID</th><th>Agent</th><th>Org</th>"
                                "<th>URL</th><th>Date</th></tr>"
                                + "".join(rows) + "</table>"
                            )
                        except Exception as e:
                            return f"<p>Error: {e}</p>"

                    refresh_btn.click(
                        _refresh_pending, [], pending_html,
                    )

                    def _approve_sub(sid):
                        try:
                            from eval_queue import update_status
                            ok = update_status(sid.strip(), "approved")
                            if ok:
                                return (
                                    f'<div style="color:#38a169">'
                                    f'Approved: {sid}</div>'
                                )
                            return (
                                f'<div style="color:#e53e3e">'
                                f'Failed to approve {sid}</div>'
                            )
                        except Exception as e:
                            return f'<div style="color:#e53e3e">{e}</div>'

                    def _reject_sub(sid):
                        try:
                            from eval_queue import update_status
                            ok = update_status(sid.strip(), "rejected")
                            if ok:
                                return (
                                    f'<div style="color:#d69e2e">'
                                    f'Rejected: {sid}</div>'
                                )
                            return (
                                f'<div style="color:#e53e3e">'
                                f'Failed to reject {sid}</div>'
                            )
                        except Exception as e:
                            return f'<div style="color:#e53e3e">{e}</div>'

                    approve_btn.click(
                        _approve_sub, [approve_id], approve_msg,
                    )
                    reject_btn.click(
                        _reject_sub, [approve_id], approve_msg,
                    )

                    def _run_dispatch(sid):
                        try:
                            import asyncio as _aio
                            from eval_queue import get_submission
                            from eval_dispatcher import dispatch_all_tasks

                            sub = get_submission(sid.strip())
                            if sub is None:
                                return (
                                    '<div style="color:#e53e3e">'
                                    'Not found</div>'
                                )
                            if sub["status"] not in (
                                "approved", "dispatching"
                            ):
                                return (
                                    f'<div style="color:#e53e3e">'
                                    f'Cannot dispatch: status='
                                    f'{sub["status"]}</div>'
                                )
                            loop = _aio.new_event_loop()
                            results = loop.run_until_complete(
                                dispatch_all_tasks(
                                    sid.strip(),
                                    sub["endpoint_url"],
                                )
                            )
                            loop.close()
                            ok = sum(
                                1 for r in results if r.get("success")
                            )
                            return (
                                f'<div style="color:#38a169">'
                                f'Dispatched: {ok}/{len(results)} '
                                f'tasks succeeded.</div>'
                            )
                        except Exception as e:
                            return f'<div style="color:#e53e3e">{e}</div>'

                    def _run_boltz(sid):
                        try:
                            from eval_queue import get_submission
                            from eval_boltz import run_boltz_posteval

                            sub = get_submission(sid.strip())
                            if sub is None:
                                return (
                                    '<div style="color:#e53e3e">'
                                    'Not found</div>'
                                )
                            per_task = json.loads(
                                sub.get("per_task_results", "{}")
                            )
                            if not per_task:
                                return (
                                    '<div style="color:#e53e3e">'
                                    "No task results to process.</div>"
                                )
                            run_boltz_posteval(per_task)
                            from eval_queue import save_task_result
                            for tid, tres in per_task.items():
                                save_task_result(sid.strip(), tid, tres)
                            return (
                                '<div style="color:#38a169">'
                                "Boltz post-assessment complete.</div>"
                            )
                        except Exception as e:
                            return f'<div style="color:#e53e3e">{e}</div>'

                    def _run_judge(sid):
                        try:
                            import eval_judge as ej
                            from eval_queue import (
                                get_submission, save_task_result, update_status,
                            )

                            sub = get_submission(sid.strip())
                            if sub is None:
                                return ('<div style="color:#e53e3e">'
                                        'Not found</div>')
                            per_task = json.loads(
                                sub.get("per_task_results", "{}")
                            )
                            if not per_task:
                                return ('<div style="color:#e53e3e">'
                                        "No task results to process.</div>")

                            update_status(sid.strip(), "scoring")
                            ej.run_judge_panel(
                                per_task,
                                agent_id=sub.get("agent_name", "unknown"),
                                dry_run=False,
                            )
                            for tid, tres in per_task.items():
                                save_task_result(sid.strip(), tid, tres)

                            n_done = sum(
                                1 for r in per_task.values()
                                if r.get("hybrid_total") is not None
                            )
                            return (
                                f'<div style="color:#38a169">'
                                f"LLM judge complete on {n_done} tasks."
                                "</div>"
                            )
                        except Exception as e:
                            import traceback
                            return (
                                f'<div style="color:#e53e3e">'
                                f'<strong>Judge error:</strong> {e}<br>'
                                f'<pre style="font-size:0.7rem">'
                                f'{traceback.format_exc()[:600]}</pre></div>'
                            )

                    def _run_finalize(sid):
                        try:
                            from eval_queue import (
                                finalize_submission,
                                get_submission,
                            )
                            from eval_scorer import aggregate_scores

                            sub = get_submission(sid.strip())
                            if sub is None:
                                return (
                                    '<div style="color:#e53e3e">'
                                    'Not found</div>'
                                )
                            per_task = json.loads(
                                sub.get("per_task_results", "{}")
                            )
                            agg = aggregate_scores(per_task)
                            finalize_submission(
                                sid.strip(),
                                overall_score=agg["overall_score"],
                                component_scores=agg["component_scores"],
                                taxonomy_scores=agg["taxonomy_scores"],
                            )
                            mode_label = agg.get("scoring_mode", "algo")
                            return (
                                f'<div style="color:#38a169">'
                                f'Finalized! Score: '
                                f'{agg["overall_score"]:.1f} '
                                f'(scoring={mode_label})</div>'
                            )
                        except Exception as e:
                            return f'<div style="color:#e53e3e">{e}</div>'

                    dispatch_btn.click(
                        _run_dispatch, [dispatch_id], pipeline_out,
                    )
                    boltz_btn.click(
                        _run_boltz, [boltz_id], pipeline_out,
                    )
                    judge_btn.click(
                        _run_judge, [judge_id], pipeline_out,
                    )
                    final_btn.click(
                        _run_finalize, [final_id], pipeline_out,
                    )

            # ══════ Tab 7: About ══════
            with gr.Tab("\u2139\ufe0f About"):
                gr.HTML(build_about())

    return app


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    create_app().launch()
