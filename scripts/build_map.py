"""Draw docs/build-map.svg: the spec's architecture, with each package colored by how much is built,
and under it every module by name.

Run from the repo root after finishing a task:

    uv run python scripts/build_map.py

It counts the real code in every module (a module with almost none is a stub), runs the tests for
the pass count, and takes who owns unfinished work from OWNERS below. Update OWNERS when tasks
move.
"""

import ast
import html
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "build-map.svg"

# Unfinished modules with an owner or a blocker, as (status, note).
OWNERS = {
    "spar-perception:autolabel/pipeline.py": ("legacy", "E5"),
    "spar-perception:autolabel/review.py": ("legacy", "E5"),
    "spar-perception:autolabel/grounding_dino.py": ("evan", "E5"),
    "spar-perception:autolabel/sam.py": ("evan", "E5"),
    "spar-perception:detect/train.py": ("evan", "E7"),
    "spar-perception:detect/infer.py": ("evan", "E7"),
    "spar-ingest:registry.py": ("evan", "E1"),
    "spar-ingest:streams.py": ("evan", "E2"),
    "spar-ingest:cut.py": ("evan", "E3"),
    "spar-ingest:events_api.py": ("blocked", "L4"),
    "spar-ingest:counts.py": ("blocked", "L4"),
    "spar-eval:stats.py": ("leo", "L14"),
    "spar-eval:inject.py": ("leo", "L14"),
    "spar-eval:score_check_sensitivity.py": ("leo", "L14"),
    "spar-eval:metrics/sim_realism.py": ("leo", "L14"),
    "spar-perception:objects.py": ("leo", "L15"),
    "spar-learn:bc.py": ("leo", "L16"),
    "spar-perception:run.py": ("leo", "L17"),
    "spar-perception:calib/find_field.py": ("leo", "L18"),
    "render:field.py": ("todo", "E6"),  # a stretch task, so not colored as assigned
    "render:camera.py": ("todo", "E6"),
}
STAGES = ["ingest", "perception", "state", "data", "sim", "learn", "eval"]
PACKAGES = ["spar-schema", *[f"spar-{s}" for s in STAGES], "spar-gt", "render"]
STATUSES = ["built", "legacy", "leo", "evan", "blocked", "todo"]

COLORS = {
    "bg": "#ffffff",
    "fg": "#1a2027",
    "muted": "#5a6572",
    "line": "#cfd6de",
    "built": "#23804f",
    "built_tint": "#e6f2eb",
    "legacy": "#a86b12",
    "leo": "#0b7f8a",
    "leo_tint": "#e3f2f3",
    "evan": "#4652c8",
    "evan_tint": "#eceefa",
    "blocked": "#c0501a",
    "todo": "#c3c9d0",
}
STYLE = """
text {{ font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; fill: {fg}; }}
.mono {{ font-family: "JetBrains Mono", Consolas, "DejaVu Sans Mono", monospace; }}
.title {{ font-size: 24px; font-weight: 700; }}
.sub {{ font-size: 14px; fill: {muted}; }}
.box {{ fill: {bg}; stroke: {line}; stroke-width: 1.2; }}
.b-built {{ fill: {built_tint}; stroke: {built}; }}
.b-partly {{ fill: {built_tint}; }}
.b-evan {{ fill: {evan_tint}; stroke: {evan}; }}
.b-leo {{ fill: {leo_tint}; stroke: {leo}; }}
.b-outside {{ fill: none; stroke: {todo}; stroke-dasharray: 4 3; }}
.n-title {{ font-size: 14px; font-weight: 600; }}
.n-line {{ font-size: 12.5px; fill: {muted}; }}
.n-pill {{ font-size: 12px; font-weight: 700; letter-spacing: 0.06em; }}
.t-built, .t-partly {{ fill: {built}; }}
.t-evan {{ fill: {evan}; }}
.t-leo {{ fill: {leo}; }}
.t-todo {{ fill: {muted}; }}
.edge {{ fill: none; stroke: {fg}; stroke-width: 1.6; opacity: 0.8; }}
.open {{ stroke-dasharray: 6 4; }}
.faint {{ opacity: 0.45; }}
.arrowhead {{ fill: {fg}; }}
.e-label {{ font-size: 11.5px; fill: {muted}; }}
.lane {{ fill: none; stroke: {line}; stroke-width: 1.2; stroke-dasharray: 2 4; }}
.lane-label {{ font-size: 12px; font-weight: 700; letter-spacing: 0.1em; fill: {muted}; }}
.legend {{ font-size: 12.5px; fill: {muted}; }}
.card {{ fill: {bg}; stroke: {line}; stroke-width: 1.2; }}
.c-title {{ font-size: 13px; font-weight: 600; }}
.c-count {{ font-size: 12px; fill: {muted}; }}
.m-name {{ font-size: 11.5px; }}
.m-todo {{ fill: {muted}; }}
.m-tag {{ font-size: 11px; font-weight: 700; fill: {muted}; }}
.h2 {{ font-size: 16px; font-weight: 700; }}
"""

TOP = 76  # height of the title band above the diagram
MAIN_X, MAIN_W, BOX_H = 350, 240, 64
ROWS = {s: 124 + 116 * i for i, s in enumerate(STAGES)}
LINES = {
    "ingest": "records, downloads, cutting",
    "perception": "labels, detect, track, calibrate",
    "state": "task state, score check, events",
    "data": "catalog, start states, exports",
    "sim": "plan-level simulator (JAX)",
    "learn": "cloning, multi-agent RL, search",
    "eval": "metrics, stats, pre-registration",
}


def code_lines(path: Path) -> int:
    body = ast.parse(path.read_text(encoding="utf-8")).body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # the module docstring
    return len({n.lineno for node in body for n in ast.walk(node) if hasattr(n, "lineno")})


def module_statuses() -> dict[str, list[tuple[str, str, str]]]:
    """Each package's modules as (path, status, note)."""
    found = {}
    for name in PACKAGES:
        src = ROOT / (
            "render/src/spar_render"
            if name == "render"
            else f"packages/{name}/src/{name.replace('-', '_')}"
        )
        modules = []
        for path in sorted(src.rglob("*.py")):
            if path.name in ("__init__.py", "cli.py"):
                continue
            rel = path.relative_to(src).as_posix()
            key = f"{name}:{rel}"
            if key in OWNERS and (OWNERS[key][0] != "todo" or code_lines(path) <= 5):
                modules.append((rel, *OWNERS[key]))
            else:
                modules.append((rel, "built" if code_lines(path) > 5 else "todo", ""))
        found[name] = modules
    return found


def statuses(modules: list[tuple[str, str, str]]) -> list[str]:
    return [status for _, status, _ in modules]


def test_summary() -> str:
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    last = run.stdout.strip().splitlines()[-1] if run.stdout.strip() else ""
    counts = re.findall(r"(\d+) (passed|failed|error)", last)
    return ", ".join(f"{n} tests {word}" for n, word in counts) or "tests not run"


def esc(text: str) -> str:
    return html.escape(text, quote=True)


def package_status(statuses: list[str]) -> str:
    built = statuses.count("built")
    if built == len(statuses):
        return "built"
    if built:
        return "partly"
    for owner in ("evan", "leo"):
        if owner in statuses:
            return owner
    return "todo"


def node(x, y, w, title, line, statuses=None, status=None, h=BOX_H, outside=False):
    tint = (
        " b-outside"
        if outside
        else f" b-{status}"
        if status in ("built", "partly", "evan", "leo")
        else ""
    )
    out = [f'<rect class="box{tint}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6"/>']
    out.append(f'<text class="n-title mono" x="{x + 14}" y="{y + 24}">{esc(title)}</text>')
    if status:
        word = {
            "built": "BUILT",
            "partly": "PARTLY BUILT",
            "evan": "EVAN",
            "leo": "NEXT: LEO",
            "todo": "NOT STARTED",
        }[status]
        out.append(
            f'<text class="n-pill t-{status}" x="{x + w - 14}" y="{y + 24}" '
            f'text-anchor="end">{word}</text>'
        )
    out.append(f'<text class="n-line" x="{x + 14}" y="{y + 43}">{esc(line)}</text>')
    if statuses:
        cursor, width = x + 14.0, w - 28
        for st in STATUSES:
            n = statuses.count(st)
            if n:
                seg = width * n / len(statuses)
                out.append(
                    f'<rect x="{cursor:.1f}" y="{y + h - 13}" width="{max(seg - 1.5, 1):.1f}" '
                    f'height="5" rx="1.5" fill="{COLORS[st]}"/>'
                )
                cursor += seg
    return "".join(out)


def arrow(points, label=None, at=(0, 0), defined=True, anchor="middle", faint=False):
    cls = "edge" + ("" if defined else " open") + (" faint" if faint else "")
    path = " ".join(f"{x},{y}" for x, y in points)
    out = f'<polyline class="{cls}" points="{path}" marker-end="url(#arrow)"/>'
    if label:
        out += (
            f'<text class="e-label mono" x="{at[0]}" y="{at[1]}" '
            f'text-anchor="{anchor}">{esc(label)}</text>'
        )
    return out


def diagram(modules: dict[str, list[tuple[str, str, str]]]) -> str:
    found = {name: statuses(rows) for name, rows in modules.items()}
    scenes = sorted(p.parent.name for p in (ROOT / "scenes").glob("*/scene.yaml"))
    parts = [
        node(
            40,
            20,
            260,
            ", ".join(f"scenes/{s}" for s in scenes),
            "rules, scoring, plugins",
            status="built",
            h=62,
        ),
        node(
            320,
            20,
            600,
            "spar-schema",
            "scenes, episode files, task state, start states, action set; every stage uses it",
            found["spar-schema"],
            package_status(found["spar-schema"]),
            h=62,
        ),
        '<rect class="lane" x="690" y="444" width="232" height="264" rx="8"/>',
        '<text class="lane-label" x="806" y="436" text-anchor="middle">OUTSIDE THIS REPO</text>',
        node(700, 466, 212, "Physics repo", "Isaac Lab, robot policy", h=60, outside=True),
        node(790, 632, 122, "VEX U robots", "team robot code", h=60, outside=True),
        node(
            40,
            240,
            200,
            "render/",
            "Blender, synthetic video",
            found["render"],
            package_status(found["render"]),
        ),
        node(
            690,
            820,
            232,
            "spar-gt",
            "scrimmage ground truth",
            found["spar-gt"],
            package_status(found["spar-gt"]),
        ),
    ]
    for stage in STAGES:
        name = f"spar-{stage}"
        parts.append(
            node(
                MAIN_X,
                ROWS[stage],
                MAIN_W,
                name,
                LINES[stage],
                found[name],
                package_status(found[name]),
            )
        )
    cx = MAIN_X + MAIN_W / 2
    hand_offs = [
        ("ingest", "cut matches (registry)", False),
        ("perception", "tracks.parquet, objects.parquet", True),
        ("state", "episodes", True),
        ("data", "episodes, start states", True),
        ("sim", "runs in-process", True),
        ("learn", "ranked candidates", True),
    ]
    for stage, label, defined in hand_offs:
        y1 = ROWS[stage] + BOX_H
        y2 = ROWS[STAGES[STAGES.index(stage) + 1]]
        parts.append(
            arrow([(cx, y1), (cx, y2 - 2)], label, (cx + 12, (y1 + y2) / 2 + 4), defined, "start")
        )
    right = MAIN_X + MAIN_W
    parts += [
        arrow([(240, 272), (MAIN_X - 2, 272)], "frames, labels", (295, 264), False),
        arrow([(140, 304), (140, 866), (MAIN_X - 2, 866)], "rendered episodes", (245, 858)),
        arrow([(right, 500), (698, 500)], "start states", (645, 492)),
        arrow([(720, 526), (720, 724), (right + 2, 724)], "rollouts", (655, 716), faint=True),
        arrow([(right, 752), (760, 752), (760, 528)], "strategies, opponents", (660, 768), False),
        arrow([(850, 526), (850, 630)], "policy", (858, 582), False, "start", faint=True),
        arrow([(850, 692), (850, 818)], "robot logs", (858, 790), False, "start", faint=True),
        arrow([(690, 852), (right + 2, 852)], "ground truth", (640, 844)),
        arrow(
            [(922, 870), (938, 870), (938, 156), (right + 2, 156)],
            "scrimmage video",
            (765, 148),
            False,
        ),
    ]
    return "".join(parts)


def legend(y: float) -> str:
    items, x = [], 40
    for color, word in [
        ("built", "Built"),
        ("legacy", "Old script"),
        ("leo", "Assigned to Leo"),
        ("evan", "Assigned to Evan"),
        ("blocked", "Waiting on others"),
        ("todo", "Not started"),
    ]:
        items.append(
            f'<rect x="{x}" y="{y - 8}" width="22" height="6" rx="2" fill="{COLORS[color]}"/>'
        )
        items.append(f'<text class="legend" x="{x + 30}" y="{y}">{word}</text>')
        x += 30 + 7.2 * len(word) + 26
    y2 = y + 26
    items.append(f'<line class="edge" x1="40" y1="{y2 - 4}" x2="68" y2="{y2 - 4}"/>')
    items.append(f'<text class="legend" x="76" y="{y2}">File format defined in spar-schema</text>')
    items.append(f'<line class="edge open" x1="330" y1="{y2 - 4}" x2="358" y2="{y2 - 4}"/>')
    items.append(f'<text class="legend" x="366" y="{y2}">Format not defined yet</text>')
    items.append(
        f'<text class="legend" x="600" y="{y2}">Bars inside each box: its modules by status</text>'
    )
    return "".join(items)


CARD_COLUMNS, CARD_GAP, LINE = 3, 16, 17
CARD_W = (960 - 80 - CARD_GAP * (CARD_COLUMNS - 1)) / CARD_COLUMNS


def module_cards(modules: dict[str, list[tuple[str, str, str]]], top: float) -> tuple[str, float]:
    """Every module by name, in a card per package, colored by status. Returns the SVG and its
    height."""
    parts = [
        f'<text class="h2" x="40" y="{top + 18}">Modules</text>',
        f'<text class="legend" x="122" y="{top + 18}">Each package\'s modules, with the task '
        "that owns the unfinished ones</text>",
    ]
    names = list(modules)
    y = top + 36
    for row in range(0, len(names), CARD_COLUMNS):
        group = names[row : row + CARD_COLUMNS]
        height = 40 + LINE * max(len(modules[name]) for name in group)
        for i, name in enumerate(group):
            x = 40 + i * (CARD_W + CARD_GAP)
            rows = modules[name]
            built = sum(status == "built" for _, status, _ in rows)
            parts.append(
                f'<rect class="card" x="{x:.1f}" y="{y}" width="{CARD_W:.1f}" height="{height}" '
                'rx="6"/>'
            )
            parts.append(
                f'<text class="c-title mono" x="{x + 12:.1f}" y="{y + 22}">{esc(name)}</text>'
            )
            parts.append(
                f'<text class="c-count" x="{x + CARD_W - 12:.1f}" y="{y + 22}" '
                f'text-anchor="end">{built} of {len(rows)}</text>'
            )
            for k, (rel, status, note) in enumerate(rows):
                ly = y + 42 + k * LINE
                faded = "" if status == "built" else " m-todo"
                parts.append(
                    f'<circle cx="{x + 17:.1f}" cy="{ly - 4}" r="4.5" fill="{COLORS[status]}"/>'
                )
                parts.append(
                    f'<text class="m-name mono{faded}" x="{x + 28:.1f}" y="{ly}">{esc(rel)}</text>'
                )
                if note:
                    parts.append(
                        f'<text class="m-tag" x="{x + CARD_W - 12:.1f}" y="{ly}" '
                        f'text-anchor="end">{esc(note)}</text>'
                    )
        y += height + CARD_GAP
    return "".join(parts), y - top


def main() -> None:
    found = module_statuses()
    total = sum(len(rows) for rows in found.values())
    built = sum(statuses(rows).count("built") for rows in found.values())
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()
    this_file = OUT.relative_to(ROOT).as_posix()
    changes = subprocess.run(
        ["git", "status", "--porcelain", "--", ".", f":(exclude){this_file}"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    where = f"commit {commit}" + (" plus uncommitted changes" if changes else "")
    summary = f"{built} of {total} modules built · {test_summary()} · {where}"
    cards, cards_height = module_cards(found, TOP + 904 + 72)
    height = TOP + 904 + 72 + cards_height + 16
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 {height}" '
        f'width="960" height="{height}" '
        f'role="img" aria-label="SPAR build map: {esc(summary)}">'
        f"<style>{STYLE.format(**COLORS)}</style>"
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        'markerHeight="7" orient="auto-start-reverse">'
        '<path class="arrowhead" d="M0,0 L10,5 L0,10 z"/></marker></defs>'
        f'<rect x="0.5" y="0.5" width="959" height="{height - 1}" rx="10" '
        f'fill="{COLORS["bg"]}" stroke="{COLORS["line"]}"/>'
        '<text class="title" x="40" y="42">SPAR build map</text>'
        f'<text class="sub" x="40" y="64">{esc(summary)}</text>'
        f'<g transform="translate(0 {TOP})">{diagram(found)}</g>'
        f"{legend(TOP + 904 + 24)}"
        f"{cards}"
        "</svg>\n"
    )
    OUT.write_text(svg, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}: {summary}")


if __name__ == "__main__":
    main()
