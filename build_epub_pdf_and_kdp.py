"""Build Publication-Grade EPUB 3.0, Digital PDF, KDP 6x9 Print Interior PDF,
and KDP Full-Wrap Paperback + Kindle eBook Covers for:
Why AI Demos Lie (First Edition, 2026) — Author: Mathurin Aché Arnaud (Kaggle Grandmaster)

Deliverables generated in `05_package/amazon_kdp_kit/` (without BOOK-002_ prefix):
  - Why_AI_Demos_Lie.epub
  - Why_AI_Demos_Lie_Digital_Edition.pdf
  - Why_AI_Demos_Lie_KDP_Print_Interior_6x9.pdf (Target: 300-420 pages, 6"x9" trim)
  - cover_kdp_ebook_1600x2560.jpg
  - cover_kdp_paperback_fullwrap_6x9.pdf & .jpg
"""

from __future__ import annotations

import html
import math
import os
import re
import shutil
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import fitz  # PyMuPDF
import markdown
import pygments
from PIL import Image, ImageDraw
from pygments.formatters import HtmlFormatter
from pygments.lexers import TextLexer, get_lexer_by_name

BOOK_DIR = Path(__file__).resolve().parent.parent
MANUSCRIPT_DIR = BOOK_DIR / "03_manuscript"
PACKAGE_DIR = BOOK_DIR / "05_package"
KDP_DIR = PACKAGE_DIR / "amazon_kdp_kit"

AUTHOR_NAME = "Mathurin Aché Arnaud"
AUTHOR_CREDENTIALS = "Kaggle Grandmaster · Top 10 World"
BOOK_TITLE = "Why AI Demos Lie"
BOOK_SUBTITLE = (
    "Statistical Evaluation, State Oracles, Fault Injection, and Post-Mortems "
    "for Production LLMs and AI Agents"
)

CHAPTER_FILES = [
    ("00_front_matter.md", "Front Matter, Notation & Companion Verification Vault"),
    ("chapters/ch01_the_pass_at_k_trap.md", "Chapter 1: The pass@k vs. pass^k Trap"),
    ("chapters/ch02_small_sample_illusions_and_confidence_intervals.md", "Chapter 2: Small-Sample Illusions & Confidence Intervals"),
    ("chapters/ch03_paired_testing_and_variance_decomposition.md", "Chapter 3: Paired Significance Testing & Variance Decomposition"),
    ("chapters/ch04_benchmark_contamination_and_data_leakage.md", "Chapter 4: Benchmark Contamination, Pre-Split Leakage & Goodhart's Law"),
    ("chapters/ch05_verifying_state_not_claims.md", "Chapter 5: Verifying State, Not Claims: Deterministic Oracles"),
    ("chapters/ch06_breaking_circular_llm_judges.md", "Chapter 6: Breaking Circular LLM Judges & Rogan-Gladen Calibration"),
    ("chapters/ch07_pairwise_ranking_elo_and_bradley_terry.md", "Chapter 7: Pairwise Ranking Systems: Bradley-Terry MLE & Cycles"),
    ("chapters/ch08_evaluating_rag_and_context_pipelines.md", "Chapter 8: Component-Level Evaluation of RAG & Context Compaction"),
    ("chapters/ch09_trajectory_evaluation_and_bounded_agency.md", "Chapter 9: Trajectory Evaluation, Tool Efficiency & Loop Termination"),
    ("chapters/ch10_latency_mathematics_and_serving_economics.md", "Chapter 10: The Mathematics of Agentic Latency & Unit Economics"),
    ("chapters/ch11_chaos_engineering_and_fault_injection.md", "Chapter 11: Chaos Engineering & Seeded Fault Injection for AI"),
    ("chapters/ch12_adversarial_red_teaming_and_boundary_probing.md", "Chapter 12: Adversarial Evaluation, Canary Oracles & Safety-Utility"),
    ("chapters/ch13_human_in_the_loop_and_annotator_reliability.md", "Chapter 13: Evaluating the Human in the Loop & Queueing Limits"),
    ("chapters/ch14_production_drift_and_continuous_shadow_evals.md", "Chapter 14: Production Drift Detection: PSI, Wasserstein & CUSUM"),
    ("chapters/ch15_forensic_tracing_and_incident_post_mortems.md", "Chapter 15: Forensic Telemetry, Deterministic Replay & Post-Mortems"),
    ("chapters/ch16_the_evaluation_release_gate.md", "Chapter 16: The Evaluation Release Gate & Versioned Manifests"),
    ("99_back_matter.md", "Appendices A-E, Rubric, Statistical Tables, FailMetry Engine & Index"),
]


# ============================================================================
# 1. RECURSIVE BALANCED-BRACE LATEX MATH PARSER (ZERO RAW BACKSLASHES)
# ============================================================================
def _extract_braced_arg(s: str, start_idx: int) -> tuple[str, int] | None:
    """Given index pointing at '{', return (inner_content, index_after_closing_brace)."""
    while start_idx < len(s) and s[start_idx] == " ":
        start_idx += 1
    if start_idx >= len(s) or s[start_idx] != "{":
        return None
    depth = 0
    chars: list[str] = []
    for i in range(start_idx, len(s)):
        ch = s[i]
        if ch == "{":
            if depth > 0:
                chars.append(ch)
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars), i + 1
            chars.append(ch)
        else:
            chars.append(ch)
    return None


def _replace_two_arg_cmd(s: str, cmd: str, formatter) -> str:
    """Replace all occurrences of \\cmd{A}{B} using balanced brace extraction."""
    while True:
        idx = s.find(cmd)
        if idx == -1:
            break
        arg1 = _extract_braced_arg(s, idx + len(cmd))
        if arg1 is None:
            break
        val1, next_idx = arg1
        arg2 = _extract_braced_arg(s, next_idx)
        if arg2 is None:
            break
        val2, end_idx = arg2
        inner1 = latex_to_clean_html(val1)
        inner2 = latex_to_clean_html(val2)
        s = s[:idx] + formatter(inner1, inner2) + s[end_idx:]
    return s


def _replace_one_arg_cmd(s: str, cmd: str, formatter) -> str:
    """Replace all occurrences of \\cmd{A} using balanced brace extraction."""
    while True:
        idx = s.find(cmd)
        if idx == -1:
            break
        arg1 = _extract_braced_arg(s, idx + len(cmd))
        if arg1 is None:
            break
        val1, end_idx = arg1
        inner1 = latex_to_clean_html(val1)
        s = s[:idx] + formatter(inner1) + s[end_idx:]
    return s


SYMBOL_MAP = [
    (r"\hat{\theta}", "θ̂"),
    (r"\hat{p}", "p̂"),
    (r"\hat{q}", "q̂"),
    (r"\hat{\beta}", "β̂"),
    (r"\hat{\pi}", "π̂"),
    (r"\hat{\mu}", "μ̂"),
    (r"\hat{\sigma}", "σ̂"),
    (r"\hat{\Delta}", "Δ̂"),
    (r"\hat{Y}", "Ŷ"),
    (r"\bar{s}", "s̄"),
    (r"\bar{x}", "x̄"),
    (r"\bar{p}", "p̄"),
    (r"\mathbb{N}^+", "ℕ⁺"),
    (r"\mathbb{N}", "ℕ"),
    (r"\mathbb{R}", "ℝ"),
    (r"\mathbb{E}", "𝔼"),
    (r"\mathbb{P}", "ℙ"),
    (r"\mathbb{I}", "𝕀"),
    (r"\mathcal{X}", "𝒳"),
    (r"\mathcal{Y}", "𝒴"),
    (r"\mathcal{S}", "𝒮"),
    (r"\mathcal{D}", "𝒟"),
    (r"\mathcal{I}", "ℐ"),
    (r"\mathcal{E}", "ℰ"),
    (r"\mathcal{F}", "ℱ"),
    (r"\mathcal{L}", "ℒ"),
    (r"\mathcal{N}", "𝒩"),
    (r"\mathcal{O}", "𝒪"),
    (r"\mathcal{A}", "𝒜"),
    (r"\mathcal{C}", "𝒞"),
    (r"\mathcal{T}", "𝒯"),
    (r"\Omega", "Ω"),
    (r"\Lambda", "Λ"),
    (r"\Phi", "Φ"),
    (r"\Psi", "Ψ"),
    (r"\Sigma", "Σ"),
    (r"\Theta", "Θ"),
    (r"\Gamma", "Γ"),
    (r"\Delta", "Δ"),
    (r"\theta", "θ"),
    (r"\lambda", "λ"),
    (r"\mu", "μ"),
    (r"\rho", "ρ"),
    (r"\sigma", "σ"),
    (r"\alpha", "α"),
    (r"\beta", "β"),
    (r"\gamma", "γ"),
    (r"\delta", "δ"),
    (r"\epsilon", "ε"),
    (r"\varepsilon", "ε"),
    (r"\zeta", "ζ"),
    (r"\eta", "η"),
    (r"\kappa", "κ"),
    (r"\nu", "ν"),
    (r"\xi", "ξ"),
    (r"\pi", "π"),
    (r"\tau", "τ"),
    (r"\phi", "φ"),
    (r"\varphi", "φ"),
    (r"\chi", "χ"),
    (r"\psi", "ψ"),
    (r"\omega", "ω"),
    (r"\ell", "ℓ"),
    (r"\nabla", "∇"),
    (r"\partial", "∂"),
    (r"\blacksquare", "■"),
    (r"\square", "□"),
    (r"\times", "×"),
    (r"\cdot", "·"),
    (r"\cdots", "⋯"),
    (r"\ldots", "…"),
    (r"\dots", "…"),
    (r"\le", "≤"),
    (r"\leq", "≤"),
    (r"\ge", "≥"),
    (r"\geq", "≥"),
    (r"\neq", "≠"),
    (r"\ne", "≠"),
    (r"\approx", "≈"),
    (r"\equiv", "≡"),
    (r"\sim", "∼"),
    (r"\simeq", "≃"),
    (r"\cong", "≅"),
    (r"\propto", "∝"),
    (r"\succ", "≻"),
    (r"\prec", "≺"),
    (r"\succeq", "⪰"),
    (r"\preceq", "⪯"),
    (r"\perp", "⊥"),
    (r"\parallel", "∥"),
    (r"\mid", " | "),
    (r"\in", "∈"),
    (r"\notin", "∉"),
    (r"\subset", "⊂"),
    (r"\subseteq", "⊆"),
    (r"\cup", "∪"),
    (r"\cap", "∩"),
    (r"\setminus", "∖"),
    (r"\emptyset", "∅"),
    (r"\to", "→"),
    (r"\rightarrow", "→"),
    (r"\leftarrow", "←"),
    (r"\leftrightarrow", "↔"),
    (r"\Rightarrow", "⇒"),
    (r"\implies", "⟹"),
    (r"\iff", "⟺"),
    (r"\mapsto", "↦"),
    (r"\pm", "±"),
    (r"\mp", "∓"),
    (r"\infty", "∞"),
    (r"\forall", "∀"),
    (r"\exists", "∃"),
    (r"\neg", "¬"),
    (r"\land", "∧"),
    (r"\lor", "∨"),
    (r"\oplus", "⊕"),
    (r"\otimes", "⊗"),
    (r"\top", "⊤"),
    (r"\bot", "⊥"),
    (r"\lfloor", "⌊"),
    (r"\rfloor", "⌋"),
    (r"\lceil", "⌈"),
    (r"\rceil", "⌉"),
    (r"\lvert", "|"),
    (r"\rvert", "|"),
    (r"\lVert", "∥"),
    (r"\rVert", "∥"),
    (r"\langle", "⟨"),
    (r"\rangle", "⟩"),
    (r"\ln", "ln"),
    (r"\log", "log"),
    (r"\exp", "exp"),
    (r"\sin", "sin"),
    (r"\cos", "cos"),
    (r"\tan", "tan"),
    (r"\min", "min"),
    (r"\max", "max"),
    (r"\sup", "sup"),
    (r"\inf", "inf"),
    (r"\arg\max", "argmax"),
    (r"\arg\min", "argmin"),
    (r"\Pr", "Pr"),
    (r"\text{clip}", "clip"),
    (r"\text{Var}", "Var"),
    (r"\text{Cov}", "Cov"),
    (r"\text{SE}", "SE"),
    (r"\left(", "("),
    (r"\right)", ")"),
    (r"\left[", "["),
    (r"\right]", "]"),
    (r"\left\{", "{"),
    (r"\right\}", "}"),
    (r"\left|", "|"),
    (r"\right|", "|"),
    (r"\bigl(", "("),
    (r"\bigr)", ")"),
    (r"\Bigl(", "("),
    (r"\Bigr)", ")"),
    (r"\big(", "("),
    (r"\big)", ")"),
    (r"\Big(", "("),
    (r"\Big)", ")"),
    (r"\bigg(", "("),
    (r"\bigg)", ")"),
    (r"\!", ""),
    (r"\quad", "  "),
    (r"\qquad", "    "),
    (r"\;", " "),
    (r"\,", " "),
    (r"\:", " "),
    (r"\|", " ∥ "),
    (r"\%", "%"),
    (r"\$", "$"),
    (r"\_", "_"),
]


def latex_to_clean_html(expr: str) -> str:
    """Recursively convert any LaTeX math expression into clean HTML/Unicode math
    with zero surviving raw LaTeX commands.
    """
    s = expr.strip()

    # Handle \xrightarrow{...}
    s = _replace_one_arg_cmd(s, r"\xrightarrow", lambda a: f" →<sup>{a}</sup> ")

    # 1. Handle 1-argument text/font wrappers first or recursively
    s = _replace_one_arg_cmd(s, r"\text", lambda a: f"<span class='m-text'>{a}</span>")
    s = _replace_one_arg_cmd(s, r"\mathrm", lambda a: f"<span class='m-text'>{a}</span>")
    s = _replace_one_arg_cmd(s, r"\operatorname", lambda a: f"<span class='m-text'>{a}</span>")
    s = _replace_one_arg_cmd(s, r"\texttt", lambda a: f"<code>{a}</code>")
    s = _replace_one_arg_cmd(s, r"\mathbf", lambda a: f"<b>{a}</b>")
    s = _replace_one_arg_cmd(s, r"\boldsymbol", lambda a: f"<b>{a}</b>")
    s = _replace_one_arg_cmd(s, r"\mathit", lambda a: f"<i>{a}</i>")
    s = _replace_one_arg_cmd(s, r"\mathcal", lambda a: f"<i>{a}</i>")
    s = _replace_one_arg_cmd(s, r"\mathbb", lambda a: f"<b>{a}</b>")
    s = _replace_one_arg_cmd(s, r"\hat", lambda a: f"{a}̂")
    s = _replace_one_arg_cmd(s, r"\widehat", lambda a: f"{a}̂")
    s = _replace_one_arg_cmd(s, r"\bar", lambda a: f"{a}̄")
    s = _replace_one_arg_cmd(s, r"\overline", lambda a: f"{a}̄")
    s = _replace_one_arg_cmd(s, r"\tilde", lambda a: f"{a}̃")
    s = _replace_one_arg_cmd(s, r"\widetilde", lambda a: f"{a}̃")
    s = _replace_one_arg_cmd(s, r"\vec", lambda a: f"{a}⃗")

    # 2. Handle 2-argument commands: \frac{A}{B} and \binom{A}{B}
    s = _replace_two_arg_cmd(
        s,
        r"\frac",
        lambda num, den: (
            f"<span class='m-frac'><span class='m-num'>({num})</span>"
            f"<span class='m-slash'>/</span><span class='m-den'>({den})</span></span>"
        ),
    )
    s = _replace_two_arg_cmd(
        s,
        r"\dfrac",
        lambda num, den: (
            f"<span class='m-frac'><span class='m-num'>({num})</span>"
            f"<span class='m-slash'>/</span><span class='m-den'>({den})</span></span>"
        ),
    )
    s = _replace_two_arg_cmd(s, r"\binom", lambda n, k: f"C({n}, {k})")

    # 3. Handle \sqrt{A}
    s = _replace_one_arg_cmd(s, r"\sqrt", lambda a: f"√({a})")

    # 4. Handle \sum, \prod, \int with limits
    s = re.sub(r"\\sum_\{([^{}]+)\}\^\{([^{}]+)\}", r"∑<sub>\1</sub><sup>\2</sup>", s)
    s = re.sub(r"\\sum_\{([^{}]+)\}", r"∑<sub>\1</sub>", s)
    s = re.sub(r"\\sum", "∑", s)
    s = re.sub(r"\\prod_\{([^{}]+)\}\^\{([^{}]+)\}", r"∏<sub>\1</sub><sup>\2</sup>", s)
    s = re.sub(r"\\prod_\{([^{}]+)\}", r"∏<sub>\1</sub>", s)
    s = re.sub(r"\\prod", "∏", s)
    s = re.sub(r"\\int_\{([^{}]+)\}\^\{([^{}]+)\}", r"∫<sub>\1</sub><sup>\2</sup>", s)
    s = re.sub(r"\\int", "∫", s)

    # 5. Apply symbol map
    for latex_tok, uni_repl in SYMBOL_MAP:
        s = s.replace(latex_tok, uni_repl)

    # 6. Handle balanced subscripts _{...} and superscripts ^{...}
    while True:
        m = re.search(r"([_^])\{", s)
        if not m:
            break
        op = m.group(1)
        brace_start = m.start() + 1
        extracted = _extract_braced_arg(s, brace_start)
        if extracted is None:
            break
        inner_val, end_idx = extracted
        tag = "sub" if op == "_" else "sup"
        s = s[: m.start()] + f"<{tag}>{latex_to_clean_html(inner_val)}</{tag}>" + s[end_idx:]

    # Simple 1-char subscripts/superscripts
    s = re.sub(r"_([a-zA-Z0-9])", r"<sub>\1</sub>", s)
    s = re.sub(r"\^([a-zA-Z0-9\+\-\*])", r"<sup>\1</sup>", s)

    # Strip any stray braces or unknown \command tokens
    s = re.sub(r"\\([a-zA-Z]+)", r"\1", s)
    s = s.replace("{", "").replace("}", "")
    return s


# ============================================================================
# 2. VECTOR SVG ARCHITECTURAL DIAGRAM RENDERER (REPLACES TEXT BOXES)
# ============================================================================
def render_mermaid_as_svg_figure(
    mermaid_code: str, fig_index: int, is_print: bool = False
) -> str:
    """Render a Mermaid flowchart/sequence/state block as a crisp, self-contained
    vector <svg> architectural diagram with nodes, subgraphs, and directed arrows.
    Adapts automatically: Dark Slate theme for Screen/EPUB, Crisp B&W for 6x9 Print.
    """
    raw_lines = [ln.strip() for ln in mermaid_code.strip().splitlines() if ln.strip()]
    diag_type = raw_lines[0] if raw_lines else "flowchart"

    nodes: list[tuple[str, str]] = []
    edges: list[tuple[str, str, str]] = []
    seen_nodes: dict[str, str] = {}

    def _clean_label(lbl: str) -> str:
        lbl = lbl.strip().strip('"').strip("'")
        lbl = lbl.replace("\\n", " | ")
        return lbl

    for ln in raw_lines[1:]:
        if ln.startswith(("end", "direction ", "autonumber", "style ", "classDef ", "class ")):
            continue
        if ln.startswith("subgraph "):
            m = re.search(r'subgraph\s+([A-Za-z0-9_]+)(?:\["?([^"\]]+)"?\])?', ln)
            if m:
                sg_id = m.group(1)
                sg_lbl = _clean_label(m.group(2) or sg_id)
                seen_nodes[sg_id] = f"GROUP: {sg_lbl}"
            continue
        if ln.startswith("participant "):
            m = re.search(r"participant\s+([A-Za-z0-9_]+)\s+as\s+(.+)", ln)
            if m:
                pid, plbl = m.group(1), _clean_label(m.group(2))
                seen_nodes[pid] = plbl
                nodes.append((pid, plbl))
            continue

        # Extract inline node definitions: ID["Label"] or ID[("Label")] or ID{"Label"} or ID[Label]
        for nm in re.finditer(
            r'([A-Za-z0-9_]+)\s*(?:\[\("?([^"\)]+)"?\)\]|\["?([^"\]]+)"?\]|\{"?([^"\}]+)"?\})',
            ln,
        ):
            nid = nm.group(1)
            nlbl = _clean_label(nm.group(2) or nm.group(3) or nm.group(4) or nid)
            if nid not in seen_nodes:
                seen_nodes[nid] = nlbl
                nodes.append((nid, nlbl))

        # Extract edges: A -->|"label"| B or A ->> B: label or A --> B: label
        em_flow = re.search(
            r'([A-Za-z0-9_\[\]\*]+)(?:[\[\{\(][^\]\}\)]*[\]\}\)])?\s*(?:-\.->|-->|==>|-->>|->>)\s*(?:\|"?([^"\|]*)"?\|)?\s*([A-Za-z0-9_\[\]\*]+)(?:[\[\{\(][^\]\}\)]*[\]\}\)])?(?:\s*:\s*(.+))?',
            ln,
        )
        if em_flow:
            src = em_flow.group(1).replace("[*]", "START")
            edge_lbl = _clean_label(em_flow.group(2) or em_flow.group(4) or "")
            dst = em_flow.group(3).replace("[*]", "END")
            for endpoint in (src, dst):
                if endpoint not in seen_nodes:
                    seen_nodes[endpoint] = endpoint
                    nodes.append((endpoint, endpoint))
            edges.append((src, dst, edge_lbl))

    items = edges[:10] if edges else [(nid, "", nlbl) for nid, nlbl in nodes[:8]]
    row_h = 44
    svg_h = max(140, 58 + len(items) * row_h)
    svg_w = 680

    if is_print:
        bg_col = "#f8fafc"
        border_col = "#cbd5e1"
        header_fill = "#0f172a"
        header_txt = "#f8fafc"
        arrow_col = "#0f172a"
        src_fill = "#e2e8f0"
        src_stroke = "#334155"
        src_txt = "#0f172a"
        lbl_fill = "#ffffff"
        lbl_stroke = "#94a3b8"
        lbl_txt = "#1e3a8a"
        dst_fill = "#dbeafe"
        dst_stroke = "#1e40af"
        dst_txt = "#0f172a"
    else:
        bg_col = "#0d1117"
        border_col = "#30363d"
        header_fill = "#161b22"
        header_txt = "#38bdf8"
        arrow_col = "#38bdf8"
        src_fill = "#1e293b"
        src_stroke = "#475569"
        src_txt = "#f1f5f9"
        lbl_fill = "#0f172a"
        lbl_stroke = "#38bdf8"
        lbl_txt = "#7dd3fc"
        dst_fill = "#0c4a6e"
        dst_stroke = "#0284c7"
        dst_txt = "#ffffff"

    svg_elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="100%" style="max-width:680px; display:block; margin: 0.9rem auto; border:1px solid {border_col}; border-radius:6px; background:{bg_col};">',
        '<defs>',
        '  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
        f'    <path d="M 0 1 L 10 5 L 0 9 z" fill="{arrow_col}"/>',
        '  </marker>',
        '</defs>',
        f'<rect x="0" y="0" width="{svg_w}" height="34" fill="{header_fill}" rx="5" ry="5"/>',
        f'<text x="16" y="22" fill="{header_txt}" font-family="Segoe UI, sans-serif" font-size="11.5" font-weight="bold">ARCHITECTURAL SCHEMATIC {fig_index} ({html.escape(diag_type.upper())})</text>',
    ]

    for idx, (src, dst, elbl) in enumerate(items):
        y_top = 46 + idx * row_h
        y_mid = y_top + 16
        src_title = seen_nodes.get(src, src)[:34]
        dst_title = seen_nodes.get(dst, dst)[:34]

        if dst:
            svg_elements.append(
                f'<rect x="16" y="{y_top}" width="215" height="30" rx="4" fill="{src_fill}" stroke="{src_stroke}" stroke-width="1.2"/>'
            )
            svg_elements.append(
                f'<text x="24" y="{y_top+19}" fill="{src_txt}" font-family="Consolas, monospace" font-size="9.5" font-weight="bold">{html.escape(src_title)}</text>'
            )
            svg_elements.append(
                f'<line x1="236" y1="{y_mid}" x2="438" y2="{y_mid}" stroke="{arrow_col}" stroke-width="1.4" marker-end="url(#arrow)"/>'
            )
            if elbl:
                svg_elements.append(
                    f'<rect x="248" y="{y_top+3}" width="178" height="15" rx="2" fill="{lbl_fill}" stroke="{lbl_stroke}" stroke-width="0.7"/>'
                )
                svg_elements.append(
                    f'<text x="337" y="{y_top+13}" text-anchor="middle" fill="{lbl_txt}" font-family="Segoe UI, sans-serif" font-size="8.5">{html.escape(elbl[:36])}</text>'
                )
            svg_elements.append(
                f'<rect x="446" y="{y_top}" width="218" height="30" rx="4" fill="{dst_fill}" stroke="{dst_stroke}" stroke-width="1.2"/>'
            )
            svg_elements.append(
                f'<text x="454" y="{y_top+19}" fill="{dst_txt}" font-family="Consolas, monospace" font-size="9.5" font-weight="bold">{html.escape(dst_title)}</text>'
            )
        else:
            svg_elements.append(
                f'<rect x="16" y="{y_top}" width="648" height="30" rx="4" fill="{src_fill}" stroke="{src_stroke}" stroke-width="1.2"/>'
            )
            svg_elements.append(
                f'<text x="28" y="{y_top+19}" fill="{src_txt}" font-family="Consolas, monospace" font-size="10" font-weight="bold">{html.escape(src_title)}: {html.escape(elbl[:75])}</text>'
            )

    svg_elements.append("</svg>")
    return "\n".join(svg_elements)


_GLOBAL_FIG_COUNTER = 0


# ============================================================================
# 3. DEVELOPER CODE HIGHLIGHTER (GITHUB DARK FOR SCREEN, CRISP B&W FOR PRINT)
# ============================================================================
def highlight_fenced_code(lang: str, code_str: str, is_print: bool = False) -> str:
    """Highlight code snippets using Pygments with inline styles (noclasses=True).
    Guarantees readable, beautiful code across all e-readers (Dark, Light, Sepia modes)
    and pristine physical rendering on 6x9 print paper.
    """
    lang = (lang or "").strip().lower()
    try:
        lexer = get_lexer_by_name(lang) if lang else TextLexer()
    except Exception:
        lexer = TextLexer()

    style_name = "bw" if is_print else "github-dark"
    formatter = HtmlFormatter(nowrap=True, noclasses=True, style=style_name)
    hl = pygments.highlight(code_str.rstrip(), lexer, formatter).strip()
    lang_lbl = lang.upper() if lang not in ("", "text") else "CODE"

    if is_print:
        return (
            f'\n\n<div class="code-window code-window-print">\n'
            f'  <div class="code-header code-header-print">\n'
            f'    <span class="code-lang-print">{lang_lbl}</span>\n'
            f'  </div>\n'
            f'  <pre class="code-body-print"><code class="code-content-print">{hl}</code></pre>\n'
            f'</div>\n\n'
        )
    else:
        return (
            f'\n\n<div class="code-window code-window-screen">\n'
            f'  <div class="code-header code-header-screen">\n'
            f'    <div class="code-dots"><span class="code-dot dot-red"></span><span class="code-dot dot-yellow"></span><span class="code-dot dot-green"></span></div>\n'
            f'    <span class="code-lang">{lang_lbl}</span>\n'
            f'  </div>\n'
            f'  <pre class="code-body-screen"><code class="code-content-screen">{hl}</code></pre>\n'
            f'</div>\n\n'
        )


def preprocess_markdown(md_text: str, is_print_interior: bool = False) -> str:
    """Clean markdown links, images, Mermaid blocks (into vector SVGs), LaTeX math,
    and syntax-highlight fenced code blocks using standard developer formatting.
    """
    global _GLOBAL_FIG_COUNTER
    text = md_text
    text = re.sub(r"!\[Front Cover[^\]]*\]\([^)]+\)\s*", "", text)

    # Convert file:/// links to clean anchor text
    text = re.sub(r"\[([^\]]+)\]\(file:///[^)]+\)", r"\1", text)

    # Convert Mermaid fenced blocks into crisp vector <svg> figures
    def _mermaid_repl(match: re.Match[str]) -> str:
        global _GLOBAL_FIG_COUNTER
        _GLOBAL_FIG_COUNTER += 1
        return render_mermaid_as_svg_figure(
            match.group(1), _GLOBAL_FIG_COUNTER, is_print=is_print_interior
        )

    text = re.sub(r"```mermaid\s*\n(.*?)```", _mermaid_repl, text, flags=re.DOTALL)

    # Protect fenced code blocks and inline code spans before math replacement
    code_blocks: list[str] = []

    def _save_code(match: re.Match[str]) -> str:
        idx = len(code_blocks)
        code_blocks.append(match.group(0))
        return f"@@CODE_BLOCK_{idx}@@"

    text = re.sub(r"```.*?```", _save_code, text, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", _save_code, text)

    # Convert display math $$...$$ and \[...\]
    def _display_math_repl(match: re.Match[str]) -> str:
        rendered = latex_to_clean_html(match.group(1))
        return f"\n<div class='math-display'>{rendered}</div>\n"

    text = re.sub(r"\$\$(.*?)\$\$", _display_math_repl, text, flags=re.DOTALL)
    text = re.sub(r"\\\[(.*?)\\\]", _display_math_repl, text, flags=re.DOTALL)

    # Convert inline math $...$ and \(...\)
    def _inline_math_repl(match: re.Match[str]) -> str:
        rendered = latex_to_clean_html(match.group(1))
        return f"<span class='math-inline'>{rendered}</span>"

    text = re.sub(r"\\\((.*?)\\\)", _inline_math_repl, text)
    text = re.sub(r"(?<!\\)\$([^\$\n]+?)\$", _inline_math_repl, text)
    text = text.replace(r"\$", "$")

    # Restore code blocks (with Pygments syntax highlighting for fenced code blocks)
    for idx, block in enumerate(code_blocks):
        if block.startswith("```"):
            m = re.match(r"^```([a-zA-Z0-9_\-]*)\s*\n(.*?)```$", block, flags=re.DOTALL)
            if m:
                lang = m.group(1)
                code_str = m.group(2)
                replacement = highlight_fenced_code(lang, code_str, is_print=is_print_interior)
                text = text.replace(f"@@CODE_BLOCK_{idx}@@", replacement)
            else:
                text = text.replace(f"@@CODE_BLOCK_{idx}@@", block)
        else:
            text = text.replace(f"@@CODE_BLOCK_{idx}@@", block)

    return text


def render_chapter_html(md_path: Path, is_print_interior: bool = False) -> str:
    raw_md = md_path.read_text(encoding="utf-8")
    cleaned_md = preprocess_markdown(raw_md, is_print_interior=is_print_interior)
    html_body = markdown.markdown(
        cleaned_md,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )
    # Verify zero unrendered LaTeX commands outside <pre>/<code> blocks
    stripped_check = re.sub(r"<pre.*?</pre>", "", html_body, flags=re.DOTALL)
    stripped_check = re.sub(r"<code.*?</code>", "", stripped_check, flags=re.DOTALL)
    for bad_token in (r"\frac", r"\text{", r"\sqrt{", r"\binom{", r"\sum_", r"\left("):
        if bad_token in stripped_check:
            # Clean any residual occurrence in prose if math delimiters were omitted
            html_body = latex_to_clean_html(html_body)
            break
    return html_body


COMMON_CSS = """
body {
    font-family: Georgia, "Times New Roman", serif;
    line-height: 1.54;
    color: #111827;
    margin: 0;
    padding: 0;
}
h1, h2, h3, h4 {
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, Helvetica, Arial, sans-serif;
    color: #0f172a;
    page-break-after: avoid;
    break-after: avoid;
}
h1 {
    font-size: 1.65rem;
    border-bottom: 2px solid #1e293b;
    padding-bottom: 0.3rem;
    margin-top: 0;
    margin-bottom: 0.85rem;
}
h2 {
    font-size: 1.25rem;
    margin-top: 1.35rem;
    margin-bottom: 0.5rem;
    color: #1e293b;
}
h3 {
    font-size: 1.05rem;
    margin-top: 1.05rem;
    margin-bottom: 0.35rem;
    color: #334155;
}
p {
    margin-top: 0;
    margin-bottom: 0.68rem;
    text-align: justify;
    orphans: 3;
    widows: 3;
}
ul, ol {
    margin-top: 0.25rem;
    margin-bottom: 0.75rem;
    padding-left: 1.35rem;
}
li {
    margin-bottom: 0.25rem;
}
img {
    max-width: 220px;
    border-radius: 8px;
    display: block;
    margin: 0.8rem 0;
}

/* ========================================================== */
/* INLINE CODE BADGE (TRANSLUCENT & ADAPTIVE TO ALL MODES)   */
/* ========================================================== */
:not(pre) > code {
    font-family: Consolas, "Courier New", monospace !important;
    font-size: 0.86em !important;
    background-color: rgba(56, 189, 248, 0.12) !important;
    color: #0284c7 !important;
    border: 1px solid rgba(56, 189, 248, 0.28) !important;
    padding: 0.08rem 0.32rem !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
}

/* ========================================================== */
/* FENCED CODE TERMINAL WINDOW (GITHUB DARK THEME)           */
/* ========================================================== */
.code-window {
    margin: 1.05rem 0 1.25rem 0 !important;
    page-break-inside: auto !important;
    break-inside: auto !important;
}
.code-window-screen {
    background-color: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.22) !important;
}
.code-header-screen {
    background-color: #161b22 !important;
    padding: 0.35rem 0.75rem !important;
    border-bottom: 1px solid #30363d !important;
    line-height: 1 !important;
    overflow: hidden !important;
}
.code-dots {
    float: left !important;
    line-height: 12px !important;
}
.code-dot {
    display: inline-block !important;
    width: 9px !important;
    height: 9px !important;
    margin-right: 5px !important;
    border-radius: 50% !important;
    vertical-align: middle !important;
}
.code-dot.dot-red { background-color: #ff5f56 !important; }
.code-dot.dot-yellow { background-color: #ffbd2e !important; }
.code-dot.dot-green { background-color: #27c93f !important; }
.code-lang {
    float: right !important;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace !important;
    font-size: 0.70rem !important;
    font-weight: 600 !important;
    color: #8b949e !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
pre.code-body-screen {
    margin: 0 !important;
    padding: 0.75rem 0.95rem !important;
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    border: none !important;
    border-radius: 0 0 6px 6px !important;
    font-family: Consolas, "Courier New", monospace !important;
    font-size: 0.76rem !important;
    line-height: 1.38 !important;
    overflow-x: auto !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    page-break-inside: auto !important;
    break-inside: auto !important;
}
pre.code-body-screen code {
    background: transparent !important;
    background-color: transparent !important;
    color: #e6edf3 !important;
    padding: 0 !important;
    border: none !important;
    font-size: inherit !important;
    font-family: inherit !important;
}

pre {
    font-family: Consolas, "Courier New", monospace !important;
    font-size: 0.76rem !important;
    line-height: 1.38 !important;
    background-color: #0d1117 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-left: 3px solid #38bdf8 !important;
    padding: 0.65rem 0.8rem !important;
    border-radius: 6px !important;
    overflow-x: auto !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    margin-bottom: 0.85rem !important;
    page-break-inside: auto !important;
    break-inside: auto !important;
}
pre code {
    background: transparent !important;
    background-color: transparent !important;
    padding: 0 !important;
    font-size: inherit !important;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 0.75rem 0 0.95rem 0;
    font-size: 0.8rem;
    page-break-inside: auto;
}
tr {
    page-break-inside: avoid;
    break-inside: avoid;
}
th, td {
    border: 1px solid #94a3b8;
    padding: 0.35rem 0.45rem;
    text-align: left;
    vertical-align: top;
}
th {
    background-color: #e2e8f0;
    font-family: "Segoe UI", sans-serif;
    font-weight: 600;
    color: #0f172a;
}
blockquote {
    margin: 0.75rem 0;
    padding: 0.55rem 0.85rem;
    background-color: #f8fafc;
    border-left: 4px solid #334155;
    font-style: italic;
    page-break-inside: avoid;
}
.math-display {
    margin: 0.65rem 0;
    padding: 0.45rem 0.75rem;
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    text-align: center;
    font-family: "Cambria Math", Georgia, serif;
    font-size: 0.98rem;
    page-break-inside: avoid;
}
.math-inline {
    font-family: "Cambria Math", Georgia, serif;
    font-style: italic;
}
.m-text {
    font-style: normal;
    font-family: Georgia, serif;
}
.m-frac {
    display: inline-block;
    vertical-align: middle;
}
.m-slash {
    padding: 0 0.12rem;
    font-weight: bold;
}
.chapter-break {
    page-break-before: always;
    break-before: page;
}

@media (prefers-color-scheme: dark) {
    body {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    h1, h2, h3, h4 {
        color: #f8fafc;
        border-bottom-color: #334155;
    }
    :not(pre) > code {
        background-color: rgba(56, 189, 248, 0.18) !important;
        color: #38bdf8 !important;
        border-color: rgba(56, 189, 248, 0.42) !important;
    }
    blockquote {
        background-color: #1e293b;
        color: #e2e8f0;
        border-left-color: #38bdf8;
    }
    th {
        background-color: #1e293b;
        color: #f8fafc;
        border-color: #475569;
    }
    td {
        border-color: #475569;
    }
    .math-display {
        background-color: #111827;
        border-color: #374151;
        color: #f3f4f6;
    }
}
"""


def stamp_author_on_front_cover(src_cover: Path, dest_cover: Path) -> None:
    """Create a 1600x2560 eBook front cover with the Author name ('MATHURIN ACHÉ ARNAUD')
    cleanly stamped on the bottom attribution band so KDP cover checks pass 100%.
    """
    img = Image.open(src_cover).convert("RGB").resize((1600, 2560), Image.Resampling.LANCZOS)
    tmp_jpg = dest_cover.with_suffix(".base.jpg")
    img.save(tmp_jpg, format="JPEG", quality=95)

    doc = fitz.open()
    page = doc.new_page(width=1600, height=2560)
    page.insert_image(fitz.Rect(0, 0, 1600, 2560), filename=str(tmp_jpg))

    banner_rect = fitz.Rect(120, 2405, 1480, 2505)
    page.draw_rect(banner_rect, color=(0.22, 0.74, 0.97), fill=(0.04, 0.08, 0.18), width=2.0)
    page.insert_textbox(
        fitz.Rect(130, 2424, 1470, 2498),
        "MATHURIN ACHÉ ARNAUD   ·   KAGGLE GRANDMASTER (TOP 10 WORLD)",
        fontsize=27,
        fontname="helv",
        color=(0.94, 0.97, 1.0),
        align=1,
    )
    pix = page.get_pixmap(dpi=300)
    pix.save(str(dest_cover))
    doc.close()
    tmp_jpg.unlink(missing_ok=True)
    final_img = Image.open(dest_cover).convert("RGB")
    final_img.save(dest_cover, format="JPEG", quality=95, dpi=(300, 300))


def build_epub(epub_path: Path, cover_jpg_path: Path) -> None:
    """Build a valid EPUB 3.0 archive with embedded cover, TOC, CSS, and XHTML5 chapters."""
    global _GLOBAL_FIG_COUNTER
    _GLOBAL_FIG_COUNTER = 0
    book_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, "why-ai-demos-lie-2026-mathurin-ache-arnaud"))
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    with zipfile.ZipFile(epub_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)

        container_xml = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"""
        zf.writestr("META-INF/container.xml", container_xml)
        zf.writestr("OEBPS/styles.css", COMMON_CSS)
        zf.write(cover_jpg_path, "OEBPS/cover.jpg")

        cover_xhtml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Cover</title>
  <style type="text/css">
    body {{ margin: 0; padding: 0; text-align: center; background-color: #080d1a; }}
    img {{ max-width: 100%; max-height: 98vh; height: auto; margin: 0 auto; }}
  </style>
</head>
<body>
  <div epub:type="cover">
    <img src="cover.jpg" alt="{BOOK_TITLE} — Front Cover"/>
  </div>
</body>
</html>"""
        zf.writestr("OEBPS/cover.xhtml", cover_xhtml)

        manifest_items = [
            '<item id="css" href="styles.css" media-type="text/css"/>',
            '<item id="cover-image" href="cover.jpg" media-type="image/jpeg" properties="cover-image"/>',
            '<item id="author-photo" href="author_photo.jpg" media-type="image/jpeg"/>',
            '<item id="cover-page" href="cover.xhtml" media-type="application/xhtml+xml"/>',
            '<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
            '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
        ]
        author_photo_src = PACKAGE_DIR / "author_photo.jpg"
        if author_photo_src.exists():
            zf.write(author_photo_src, "OEBPS/author_photo.jpg")
        spine_items = ['<itemref idref="cover-page"/>']
        nav_li_items = []
        ncx_navpoints = []

        for idx, (fname, title) in enumerate(CHAPTER_FILES):
            chap_id = f"chap_{idx:02d}"
            xhtml_name = f"{chap_id}.xhtml"
            html_fragment = render_chapter_html(MANUSCRIPT_DIR / fname, is_print_interior=False)
            html_fragment = re.sub(r"<hr\s*>", "<hr/>", html_fragment)
            html_fragment = re.sub(r"<br\s*>", "<br/>", html_fragment)
            html_fragment = re.sub(r'<img([^>]+?)(?<!/)>', r'<img\1/>', html_fragment)

            chap_xhtml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{html.escape(title)}</title>
  <link rel="stylesheet" type="text/css" href="styles.css"/>
</head>
<body>
{html_fragment}
</body>
</html>"""
            zf.writestr(f"OEBPS/{xhtml_name}", chap_xhtml)
            manifest_items.append(
                f'<item id="{chap_id}" href="{xhtml_name}" media-type="application/xhtml+xml" properties="svg"/>'
                if "<svg" in html_fragment
                else f'<item id="{chap_id}" href="{xhtml_name}" media-type="application/xhtml+xml"/>'
            )
            spine_items.append(f'<itemref idref="{chap_id}"/>')
            nav_li_items.append(f'<li><a href="{xhtml_name}">{html.escape(title)}</a></li>')
            ncx_navpoints.append(
                f"""    <navPoint id="navPoint-{idx+1}" playOrder="{idx+1}">
      <navLabel><text>{html.escape(title)}</text></navLabel>
      <content src="{xhtml_name}"/>
    </navPoint>"""
            )

        nav_xhtml = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Table of Contents</title>
  <link rel="stylesheet" type="text/css" href="styles.css"/>
</head>
<body>
  <nav epub:type="toc" id="toc">
    <h1>Table of Contents</h1>
    <ol>
      {"\n      ".join(nav_li_items)}
    </ol>
  </nav>
</body>
</html>"""
        zf.writestr("OEBPS/nav.xhtml", nav_xhtml)

        toc_ncx = f"""<?xml version="1.0" encoding="UTF-8"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <head>
    <meta name="dtb:uid" content="urn:uuid:{book_uuid}"/>
    <meta name="dtb:depth" content="1"/>
    <meta name="dtb:totalPageCount" content="0"/>
    <meta name="dtb:maxPageNumber" content="0"/>
  </head>
  <docTitle><text>{BOOK_TITLE}: {BOOK_SUBTITLE}</text></docTitle>
  <docAuthor><text>{AUTHOR_NAME}</text></docAuthor>
  <navMap>
{"\n".join(ncx_navpoints)}
  </navMap>
</ncx>"""
        zf.writestr("OEBPS/toc.ncx", toc_ncx)

        content_opf = f"""<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="BookId">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="BookId">urn:uuid:{book_uuid}</dc:identifier>
    <dc:title>{BOOK_TITLE}: {BOOK_SUBTITLE}</dc:title>
    <dc:language>en</dc:language>
    <dc:creator>{AUTHOR_NAME}</dc:creator>
    <dc:publisher>MSDW Systems Engineering Press</dc:publisher>
    <dc:description>Written by Kaggle Grandmaster Mathurin Aché Arnaud, this rigorous engineering reference replaces AI demo theatre and uncalibrated LLM judges with Wilson/Clopper-Pearson bounds, paired McNemar tests, Rogan-Gladen bias correction, Bradley-Terry arenas, deterministic state-mutation oracles, chaos fault injection, and 4-stage production release gates.</dc:description>
    <dc:date>2026-10-25</dc:date>
    <meta property="dcterms:modified">{now_iso}</meta>
    <meta name="cover" content="cover-image"/>
  </metadata>
  <manifest>
    {"\n    ".join(manifest_items)}
  </manifest>
  <spine toc="ncx">
    {"\n    ".join(spine_items)}
  </spine>
</package>"""
        zf.writestr("OEBPS/content.opf", content_opf)


def build_combined_html_for_pdf(
    html_out_path: Path, is_print_interior_6x9: bool, cover_jpg_path: Path
) -> None:
    """Generate a single print-ready HTML document for Edge Headless PDF rendering."""
    global _GLOBAL_FIG_COUNTER
    _GLOBAL_FIG_COUNTER = 0
    sections_html = []
    if not is_print_interior_6x9:
        cover_uri = cover_jpg_path.resolve().as_uri()
        sections_html.append(
            f"""<div style="page-break-after: always; text-align: center; padding-top: 6mm;">
  <img src="{cover_uri}" style="max-width: 92%; max-height: 245mm; border: 1px solid #1e293b; margin: 0 auto;" alt="Front Cover"/>
</div>"""
        )

    author_uri = (PACKAGE_DIR / "author_photo.jpg").resolve().as_uri()
    part_openers = {
        "chapters/ch01_the_pass_at_k_trap.md": (
            "PART I",
            "The Anatomy of the Demo Illusion",
            "Probability, Compounding Trajectories, Paired Inference, and Benchmark Contamination",
        ),
        "chapters/ch05_verifying_state_not_claims.md": (
            "PART II",
            "Beyond Circular Judges",
            "Deterministic State Oracles, Rogan-Gladen Calibrated Evaluators, Pairwise Arenas, and RAG Diagnostics",
        ),
        "chapters/ch09_trajectory_evaluation_and_bounded_agency.md": (
            "PART III",
            "Breaking the System Before Users Do",
            "Trajectory Auditing, Tail Latency Economics, Chaos Fault Injection, and Adversarial Canary Oracles",
        ),
        "chapters/ch13_human_in_the_loop_and_annotator_reliability.md": (
            "PART IV",
            "Production Telemetry, Drift & Governance",
            "Human-in-the-Loop Queueing Math, PSI/CUSUM Drift Monitors, Forensic Replay, and the 4-Stage Release Gate",
        ),
    }

    for idx, (fname, _title) in enumerate(CHAPTER_FILES):
        if fname in part_openers:
            p_num, p_title, p_sub = part_openers[fname]
            sections_html.append(
                f"""<section class="chapter-section chapter-break" style="padding-top: 2.1in; text-align: center;">
  <div style="font-family: 'Segoe UI', sans-serif; font-size: 13pt; font-weight: 700; letter-spacing: 0.18em; color: #334155; text-transform: uppercase; margin-bottom: 0.22in;">{p_num}</div>
  <h1 style="font-size: 22pt; border-bottom: 3px solid #0f172a; display: inline-block; padding-bottom: 0.14in; margin-bottom: 0.22in;">{p_title}</h1>
  <p style="text-align: center; font-size: 11.5pt; color: #475569; max-width: 4.1in; margin: 0 auto; font-style: italic;">{p_sub}</p>
</section>"""
            )

        chap_html = render_chapter_html(
            MANUSCRIPT_DIR / fname, is_print_interior=is_print_interior_6x9
        )
        chap_html = chap_html.replace('src="author_photo.jpg"', f'src="{author_uri}"')
        if fname == "99_back_matter.md":
            for app_hdr in (
                "<h2 id=\"appendix-b",
                "<h2 id=\"appendix-c",
                "<h2 id=\"appendix-d",
                "<h2 id=\"appendix-e",
                "<h2 id=\"comprehensive-subject",
                "<h2 id=\"about-the-author",
            ):
                chap_html = chap_html.replace(
                    app_hdr, f'<div class="chapter-break"></div>{app_hdr}'
                )
        cls = "chapter-section" if idx == 0 else "chapter-section chapter-break"
        sections_html.append(f'<section class="{cls}">\n{chap_html}\n</section>')

    if is_print_interior_6x9:
        page_css = """
@page {
    size: 6in 9in;
    margin-top: 0.78in;
    margin-bottom: 0.78in;
    margin-left: 0.80in;
    margin-right: 0.80in;
}
body {
    font-size: 11.4pt;
    line-height: 1.66;
}
h1 { font-size: 17.5pt; margin-top: 0.20in; margin-bottom: 0.18in; }
h2 { font-size: 13.5pt; margin-top: 0.26in; margin-bottom: 0.11in; }
h3 { font-size: 11.6pt; margin-top: 0.18in; margin-bottom: 0.08in; }
p { margin-bottom: 0.14in; }
li { margin-bottom: 0.06in; }
table { font-size: 8.8pt; margin: 0.14in 0 0.18in 0; }
th, td { padding: 0.42rem 0.52rem; }
.math-display { font-size: 10.8pt; padding: 0.60rem 0.85rem; margin: 0.85rem 0; }

/* 6x9 Print-Specific Monochrome Code Styling */
.code-window-print {
    background-color: #f8fafc !important;
    border: 1px solid #94a3b8 !important;
    border-left: 3.5px solid #0f172a !important;
    border-radius: 4px !important;
    margin: 0.80rem 0 1.00rem 0 !important;
    page-break-inside: auto !important;
    break-inside: auto !important;
}
.code-header-print {
    background-color: #e2e8f0 !important;
    padding: 0.22rem 0.58rem !important;
    border-bottom: 1px solid #cbd5e1 !important;
    text-align: right !important;
}
.code-lang-print {
    font-family: Consolas, "Courier New", monospace !important;
    font-size: 7.5pt !important;
    font-weight: 700 !important;
    color: #1e293b !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
pre.code-body-print {
    margin: 0 !important;
    padding: 0.55rem 0.68rem !important;
    background-color: #f8fafc !important;
    color: #0f172a !important;
    border: none !important;
    font-family: Consolas, "Courier New", monospace !important;
    font-size: 8.5pt !important;
    line-height: 1.44 !important;
    white-space: pre-wrap !important;
    word-wrap: break-word !important;
    page-break-inside: auto !important;
    break-inside: auto !important;
}
pre.code-body-print code {
    background: transparent !important;
    background-color: transparent !important;
    color: #0f172a !important;
    padding: 0 !important;
    border: none !important;
    font-size: inherit !important;
    font-family: inherit !important;
}
:not(pre) > code {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    border: 1px solid #cbd5e1 !important;
    padding: 0.05rem 0.24rem !important;
    font-size: 9.4pt !important;
}
"""
    else:
        page_css = """
@page {
    size: A4;
    margin: 18mm 18mm 20mm 18mm;
}
body {
    font-size: 10.5pt;
    line-height: 1.54;
}
"""

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>{BOOK_TITLE} — {AUTHOR_NAME}</title>
  <style>
{COMMON_CSS}
{page_css}
  </style>
</head>
<body>
{"\n".join(sections_html)}
</body>
</html>"""
    html_out_path.write_text(full_html, encoding="utf-8")


def render_pdf_via_edge(html_path: Path, pdf_path: Path, is_6x9: bool = False) -> None:
    from playwright.sync_api import sync_playwright

    edge_exe = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    if not Path(edge_exe).exists():
        edge_exe = r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=edge_exe)
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="load")
        if is_6x9:
            page.pdf(
                path=str(pdf_path),
                width="6in",
                height="9in",
                print_background=True,
                margin={
                    "top": "0.75in",
                    "bottom": "0.75in",
                    "left": "0.78in",
                    "right": "0.78in",
                },
            )
        else:
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={
                    "top": "18mm",
                    "bottom": "20mm",
                    "left": "18mm",
                    "right": "18mm",
                },
            )
        browser.close()


def stamp_headers_metadata_and_page_numbers(pdf_path: Path, is_6x9: bool) -> int:
    """Stamp running headers, bottom page numbers, and complete PDF Author/Title metadata."""
    doc = fitz.open(pdf_path)
    doc.set_metadata(
        {
            "title": f"{BOOK_TITLE}: {BOOK_SUBTITLE}",
            "author": AUTHOR_NAME,
            "subject": "Statistical Evaluation, State Oracles, Fault Injection & Post-Mortems for Production AI",
            "keywords": "AI Evaluation, LLM-as-a-Judge, Rogan-Gladen, Bradley-Terry, State Oracles, Chaos Engineering, Python",
            "creator": "MSDW Systems Engineering Press",
        }
    )

    total_pages = len(doc)
    start_page_idx = 1 if is_6x9 else 2
    for i in range(start_page_idx, total_pages):
        page = doc[i]
        rect = page.rect
        page_num_str = str(i + 1)
        if i > start_page_idx:
            header_text = (
                "WHY AI DEMOS LIE: STATISTICAL EVALUATION & STATE ORACLES"
                if (i % 2 == 1)
                else f"{AUTHOR_NAME.upper()}  ·  KAGGLE GRANDMASTER"
            )
            page.insert_text(
                fitz.Point(rect.width / 2 - 110, 28),
                header_text,
                fontsize=7.0,
                fontname="helv",
                color=(0.35, 0.4, 0.48),
            )
        page.insert_text(
            fitz.Point(rect.width / 2 - 8, rect.height - 26),
            page_num_str,
            fontsize=8.5,
            fontname="helv",
            color=(0.2, 0.25, 0.33),
        )

    if is_6x9 and (len(doc) % 2 == 1):
        doc.new_page(width=doc[0].rect.width, height=doc[0].rect.height)

    final_count = len(doc)
    tmp_path = pdf_path.with_suffix(".stamped.pdf")
    doc.save(tmp_path, deflate=True)
    doc.close()
    shutil.move(str(tmp_path), str(pdf_path))
    return final_count


def build_kdp_covers(
    stamped_ebook_cover: Path,
    paperback_pdf_out: Path,
    paperback_jpg_out: Path,
    page_count_6x9: int,
) -> dict[str, float]:
    """Create KDP 6" x 9" Full-Wrap Paperback Cover PDF & JPEG with Author attribution."""
    src_img = Image.open(stamped_ebook_cover).convert("RGB")

    dpi = 300
    bleed_in = 0.125
    trim_w_in = 6.0
    trim_h_in = 9.0
    spine_w_in = page_count_6x9 * 0.002252
    full_w_in = bleed_in + trim_w_in + spine_w_in + trim_w_in + bleed_in
    full_h_in = bleed_in + trim_h_in + bleed_in

    full_w_px = int(round(full_w_in * dpi))
    full_h_px = int(round(full_h_in * dpi))
    bleed_px = int(round(bleed_in * dpi))
    trim_w_px = int(round(trim_w_in * dpi))
    spine_w_px = int(round(spine_w_in * dpi))

    canvas = Image.new("RGB", (full_w_px, full_h_px), (8, 13, 26))
    draw = ImageDraw.Draw(canvas)

    grid_color = (18, 30, 58)
    for x in range(0, bleed_px + trim_w_px + spine_w_px, 60):
        draw.line([(x, 0), (x, full_h_px)], fill=grid_color, width=1)
    for y in range(0, full_h_px, 60):
        draw.line([(0, y), (bleed_px + trim_w_px + spine_w_px, y)], fill=grid_color, width=1)

    front_x = bleed_px + trim_w_px + spine_w_px
    front_w_px = full_w_px - front_x
    front_resized = src_img.resize((front_w_px, full_h_px), Image.Resampling.LANCZOS)
    canvas.paste(front_resized, (front_x, 0))

    spine_left = bleed_px + trim_w_px
    spine_right = spine_left + spine_w_px
    draw.line([(spine_left, 0), (spine_left, full_h_px)], fill=(56, 189, 248), width=2)
    draw.line([(spine_right, 0), (spine_right, full_h_px)], fill=(56, 189, 248), width=2)

    canvas.save(paperback_jpg_out, format="JPEG", quality=95, dpi=(300, 300))

    doc = fitz.open()
    width_pt = full_w_in * 72.0
    height_pt = full_h_in * 72.0
    page = doc.new_page(width=width_pt, height=height_pt)
    page.insert_image(fitz.Rect(0, 0, width_pt, height_pt), filename=str(paperback_jpg_out))

    left_margin_pt = (bleed_in + 0.55) * 72.0
    right_limit_pt = (bleed_in + trim_w_in - 0.55) * 72.0

    page.insert_textbox(
        fitz.Rect(left_margin_pt, 48, right_limit_pt, 112),
        "THE PUBLIC LEADERBOARD IS A LIAR.\nSTOP SHIPPING AI SYSTEMS ON DEMO THEATRE.",
        fontsize=12.0,
        fontname="helv",
        color=(0.22, 0.74, 0.97),
    )

    blurb_text = (
        "Every week, enterprise AI teams celebrate a 90% score on a cherry-picked 50-prompt demo "
        "graded by an uncalibrated LLM-as-a-Judge—only to watch their autonomous agent collapse "
        "in production when trajectories compound, APIs return HTTP 429s, or retrieved context drifts.\n\n"
        "Written by Kaggle Grandmaster (Global Peak Rank #6 out of 20M+ Data Scientists) and Senior "
        "Data Scientist Mathurin Aché Arnaud, Why AI Demos Lie replaces vibe-based evaluation with the "
        "rigorous mathematical and software engineering discipline of Failure-Mode Telemetry (FailMetry).\n\n"
        "INSIDE THIS 16-CHAPTER FIELD MANUAL & PYTHON VERIFICATION VAULT:\n"
        "• The Anatomy of the Lie: Compound trajectory decay (p^k), MinHash & AST contamination "
        "detection, counterfactual perturbation harnesses, Goodhart's Law, and Simpson's Paradox.\n"
        "• Statistical Rigor: Wilson score and Clopper-Pearson exact bounds, paired McNemar tests "
        "that cut sample sizes by 4x, and Holm-Bonferroni multiple-testing control.\n"
        "• Truth Engines: De-biasing LLM judges via the Rogan-Gladen estimator and delta-method "
        "variance, Zermelo Bradley-Terry arena solvers, and claim-bipartite RAG attribution.\n"
        "• Verifying State, Not Strings: Deterministic SHA-256 state-mutation oracles Omega(s0, sT) "
        "and why pass@k flatters demos while pass^k measures production reality.\n"
        "• Chaos, Drift & Governance: Seeded fault-injection proxies, OWASP red-teaming, PSI/CUSUM "
        "drift monitors, Wald SPRT canary rollbacks, Fault-Tree post-mortems, and the 4-Stage Release Gate."
    )
    page.insert_textbox(
        fitz.Rect(left_margin_pt, 110, right_limit_pt, 462),
        blurb_text,
        fontsize=8.8,
        fontname="helv",
        color=(0.91, 0.95, 0.99),
    )

    vault_rect = fitz.Rect(left_margin_pt, 466, right_limit_pt, 526)
    page.draw_rect(vault_rect, color=(0.22, 0.74, 0.97), fill=(0.07, 0.14, 0.28), width=1.0)
    page.insert_textbox(
        fitz.Rect(left_margin_pt + 10, 473, right_limit_pt - 10, 522),
        "INCLUDES 17 ZERO-DEPENDENCY PYTHON 3.12+ VERIFICATION MODULES\n"
        "Author: Mathurin Aché Arnaud (Kaggle Grandmaster)  ·  Key: DEMO-TRUTH-2026-SHA256\n"
        "Companion Volume to: The Architecture Is the Guardrail (BOOK-001)",
        fontsize=7.6,
        fontname="cour",
        color=(0.65, 0.92, 0.99),
    )

    bc_right = (bleed_in + trim_w_in - 0.40) * 72.0
    bc_left = bc_right - (2.0 * 72.0)
    bc_bottom = (full_h_in - bleed_in - 0.40) * 72.0
    bc_top = bc_bottom - (1.2 * 72.0)
    bc_rect = fitz.Rect(bc_left, bc_top, bc_right, bc_bottom)
    page.draw_rect(bc_rect, color=(0.8, 0.8, 0.8), fill=(1.0, 1.0, 1.0), width=0.5)
    page.insert_textbox(
        fitz.Rect(bc_left + 8, bc_top + 28, bc_right - 8, bc_bottom - 10),
        "AMAZON KDP ISBN BARCODE ZONE\n(2.0\" x 1.2\" White Clearance)",
        fontsize=7.5,
        fontname="helv",
        color=(0.35, 0.35, 0.35),
        align=1,
    )

    spine_center_x = (bleed_in + trim_w_in + spine_w_in / 2.0) * 72.0
    page.insert_text(
        fitz.Point(spine_center_x - 3.5, 115),
        f"WHY AI DEMOS LIE   —   {AUTHOR_NAME.upper()}",
        fontsize=9.0,
        fontname="helv",
        color=(0.95, 0.97, 1.0),
        rotate=270,
    )

    doc.set_metadata({"title": f"{BOOK_TITLE} — Paperback Cover", "author": AUTHOR_NAME})
    doc.save(paperback_pdf_out, deflate=True)
    pix = page.get_pixmap(dpi=300)
    pix.save(str(paperback_jpg_out))
    doc.close()

    return {
        "page_count_6x9": page_count_6x9,
        "spine_width_in": round(spine_w_in, 4),
        "full_width_in": round(full_w_in, 4),
        "full_height_in": round(full_h_in, 4),
    }


def main() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    KDP_DIR.mkdir(parents=True, exist_ok=True)

    for old_f in KDP_DIR.glob("BOOK-002_*"):
        old_f.unlink(missing_ok=True)

    cover_src = PACKAGE_DIR / "cover_front.jpg"
    ebook_cover_out = KDP_DIR / "cover_kdp_ebook_1600x2560.jpg"
    print("0. Stamping Author Attribution on 1600x2560 Front Cover...")
    stamp_author_on_front_cover(cover_src, ebook_cover_out)

    epub_kdp = KDP_DIR / "Why_AI_Demos_Lie.epub"
    digital_pdf_kdp = KDP_DIR / "Why_AI_Demos_Lie_Digital_Edition.pdf"
    kdp_interior_pdf_out = KDP_DIR / "Why_AI_Demos_Lie_KDP_Print_Interior_6x9.pdf"

    print("1. Building EPUB 3.0 package (with SVG diagrams & clean math)...")
    build_epub(epub_kdp, ebook_cover_out)
    shutil.copy2(epub_kdp, PACKAGE_DIR / "Why_AI_Demos_Lie.epub")
    print(f"   [OK] EPUB written to: {epub_kdp} ({epub_kdp.stat().st_size // 1024} KB)")

    print("2. Building Digital Edition PDF (A4 with Stamped Front Cover & SVG Figures)...")
    digital_html = PACKAGE_DIR / "_tmp_digital.html"
    build_combined_html_for_pdf(digital_html, is_print_interior_6x9=False, cover_jpg_path=ebook_cover_out)
    render_pdf_via_edge(digital_html, digital_pdf_kdp, is_6x9=False)
    digital_pages = stamp_headers_metadata_and_page_numbers(digital_pdf_kdp, is_6x9=False)
    digital_html.unlink(missing_ok=True)
    shutil.copy2(digital_pdf_kdp, PACKAGE_DIR / "Why_AI_Demos_Lie_Digital_Edition.pdf")
    print(f"   [OK] Digital PDF written to: {digital_pdf_kdp} ({digital_pages} pages)")

    print("3. Building Amazon KDP 6x9 Print Interior PDF...")
    kdp_html = PACKAGE_DIR / "_tmp_kdp_6x9.html"
    build_combined_html_for_pdf(kdp_html, is_print_interior_6x9=True, cover_jpg_path=ebook_cover_out)
    render_pdf_via_edge(kdp_html, kdp_interior_pdf_out, is_6x9=True)
    kdp_pages = stamp_headers_metadata_and_page_numbers(kdp_interior_pdf_out, is_6x9=True)
    kdp_html.unlink(missing_ok=True)
    shutil.copy2(kdp_interior_pdf_out, PACKAGE_DIR / "Why_AI_Demos_Lie_KDP_Print_Interior_6x9.pdf")
    print(f"   [OK] KDP 6x9 Interior PDF written to: {kdp_interior_pdf_out} ({kdp_pages} pages)")

    print("4. Building Amazon KDP eBook & Full-Wrap Paperback Covers...")
    paperback_pdf_out = KDP_DIR / "cover_kdp_paperback_fullwrap_6x9.pdf"
    paperback_jpg_out = KDP_DIR / "cover_kdp_paperback_fullwrap_6x9.jpg"
    dims = build_kdp_covers(
        stamped_ebook_cover=ebook_cover_out,
        paperback_pdf_out=paperback_pdf_out,
        paperback_jpg_out=paperback_jpg_out,
        page_count_6x9=kdp_pages,
    )
    print(f"   [OK] KDP Covers built: {dims}")


if __name__ == "__main__":
    main()
