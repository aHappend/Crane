from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Any


def _fmt_num(x: float) -> str:
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-3):
        return f"{x:.3e}"
    return f"{x:.6f}".rstrip("0").rstrip(".")


def _delta_from_cumulative(sct: list[list[float]]) -> list[list[float]]:
    if not sct:
        return []
    out: list[list[float]] = []
    prev = [0.0 for _ in sct[0]]
    for row in sct:
        out_row = [max(0.0, float(v) - float(p)) for v, p in zip(row, prev)]
        out.append(out_row)
        prev = list(row)
    return out


def _matrix_html(title: str, rows: list[str], cols: list[str], mat: list[list[float]]) -> str:
    if not mat:
        return f"<h3>{escape(title)}</h3><p>(empty)</p>"
    vmax = max((max(r) if r else 0.0) for r in mat)
    vmax = max(vmax, 1e-9)

    head = "".join(f"<th>{escape(c)}</th>" for c in cols)
    body_parts: list[str] = []
    for i, row in enumerate(mat):
        cells: list[str] = []
        for v in row:
            ratio = max(0.0, min(1.0, float(v) / vmax))
            shade = int(245 - 130 * ratio)
            bg = f"rgb(230,{shade},{shade})"
            cells.append(f"<td style='background:{bg}'>{_fmt_num(float(v))}</td>")
        body_parts.append(f"<tr><th>{escape(rows[i])}</th>{''.join(cells)}</tr>")

    return (
        f"<h3>{escape(title)}</h3>"
        "<div class='table-wrap'><table>"
        f"<thead><tr><th>State</th>{head}</tr></thead>"
        f"<tbody>{''.join(body_parts)}</tbody>"
        "</table></div>"
    )


def write_schedule_html(
    output_path: str | Path,
    *,
    title: str,
    meta: dict[str, Any],
    scheduled_blocks: list[str],
    state_order: list[str],
    state_categories: list[str],
    state_batches: list[int],
    state_active_blocks: list[list[int]],
    sct: list[list[float]],
    met_s: list[list[float]],
    met_d: list[list[float]],
    hierarchy_notes: list[str] | None = None,
    hierarchy_traces: list[dict[str, Any]] | None = None,
) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    delta = _delta_from_cumulative(sct)

    # Summary cards
    meta_html = "".join(
        f"<div class='kv'><span>{escape(str(k))}</span><b>{escape(str(v))}</b></div>"
        for k, v in meta.items()
    )

    # State table
    state_rows: list[str] = []
    for i, s in enumerate(state_order):
        active_ids = state_active_blocks[i] if i < len(state_active_blocks) else []
        active_names = [scheduled_blocks[j] for j in active_ids if 0 <= j < len(scheduled_blocks)]
        state_rows.append(
            "<tr>"
            f"<td>{escape(s)}</td>"
            f"<td>{escape(state_categories[i] if i < len(state_categories) else '')}</td>"
            f"<td>{int(state_batches[i]) if i < len(state_batches) else 0}</td>"
            f"<td>{escape(', '.join(active_names[:10]))}{' ...' if len(active_names) > 10 else ''}</td>"
            "</tr>"
        )

    state_table = (
        "<h3>State Overview</h3>"
        "<div class='table-wrap'><table><thead><tr>"
        "<th>State</th><th>Category</th><th>Assigned Sub-batches</th><th>Active Blocks</th>"
        "</tr></thead><tbody>"
        f"{''.join(state_rows)}"
        "</tbody></table></div>"
    )

    # Block cards (inter + intra)
    block_cards: list[str] = []
    for bi, bname in enumerate(scheduled_blocks):
        layers = [x for x in bname.split('|') if x]
        timeline = []
        for si, sname in enumerate(state_order):
            d = float(delta[si][bi]) if si < len(delta) and bi < len(delta[si]) else 0.0
            if d > 0:
                timeline.append(f"<span class='badge'>{escape(sname)}:+{_fmt_num(d)}</span>")
        timeline_html = " ".join(timeline) if timeline else "<span class='muted'>No processing</span>"

        if len(layers) > 120:
            layer_view = ", ".join(layers[:40]) + " ... " + ", ".join(layers[-20:])
        else:
            layer_view = ", ".join(layers)

        block_cards.append(
            "<div class='card'>"
            f"<h4>Block[{bi}] {escape(bname)}</h4>"
            f"<p><b>Inner layers ({len(layers)}):</b> {escape(layer_view)}</p>"
            f"<p><b>State schedule:</b> {timeline_html}</p>"
            "</div>"
        )

    hierarchy_html = ""
    if hierarchy_traces:
        cards = []
        for tr in hierarchy_traces[:50]:
            path = str(tr.get('path', ''))
            level = tr.get('level', '')
            sb = tr.get('best_sub_batch', '')
            cards.append(
                "<div class='card'>"
                f"<h4>Trace L{escape(str(level))} {escape(path)}</h4>"
                f"<p>best_sub_batch={escape(str(sb))}, blocks={len(tr.get('scheduled_blocks', []))}, states={len(tr.get('state_order', []))}</p>"
                "</div>"
            )
        hierarchy_html = "<h3>Hierarchy Traces</h3>" + "".join(cards)

    notes_html = ""
    if hierarchy_notes:
        lines = "".join(f"<li>{escape(line)}</li>" for line in hierarchy_notes[:300])
        notes_html = f"<h3>Hierarchy Notes</h3><ul>{lines}</ul>"

    html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>{escape(title)}</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 18px; color: #1f2937; background: linear-gradient(160deg,#f6fbff,#eef7f3); }}
h1 {{ margin: 0 0 12px 0; }}
h2 {{ margin: 18px 0 8px 0; }}
h3 {{ margin: 14px 0 6px 0; }}
.grid {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(240px,1fr)); gap:8px; }}
.kv {{ background:#fff; border:1px solid #d9e2ec; border-radius:8px; padding:8px 10px; display:flex; justify-content:space-between; gap:8px; }}
.kv span {{ color:#52606d; }}
.table-wrap {{ overflow:auto; border:1px solid #d9e2ec; border-radius:8px; background:#fff; }}
table {{ border-collapse:collapse; width:100%; min-width:640px; }}
th,td {{ border:1px solid #e5e7eb; padding:6px 8px; text-align:center; font-size:12px; }}
th {{ background:#f3f4f6; }}
.card {{ background:#fff; border:1px solid #d9e2ec; border-radius:8px; padding:10px; margin:8px 0; }}
.badge {{ display:inline-block; border:1px solid #93c5fd; background:#dbeafe; border-radius:999px; padding:2px 8px; margin:2px 4px 2px 0; font-size:12px; }}
.muted {{ color:#6b7280; }}
</style>
</head>
<body>
<h1>{escape(title)}</h1>
<h2>Run Meta</h2>
<div class='grid'>{meta_html}</div>
{state_table}
<h2>Inter-layer Scheduling</h2>
{_matrix_html('ScT cumulative (state x block)', state_order, [f'B{{i}}' for i in range(len(scheduled_blocks))], sct)}
{_matrix_html('ScT delta per state (state x block)', state_order, [f'B{{i}}' for i in range(len(scheduled_blocks))], delta)}
{_matrix_html('MeT_S (state x block)', state_order, [f'B{{i}}' for i in range(len(scheduled_blocks))], met_s)}
{_matrix_html('MeT_D (state x block)', state_order, [f'B{{i}}' for i in range(len(scheduled_blocks))], met_d)}
<h2>Intra-block (Layer-in-Block) View</h2>
{''.join(block_cards)}
{hierarchy_html}
{notes_html}
</body>
</html>
"""

    out.write_text(html, encoding='utf-8')
