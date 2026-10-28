"""Local Multi-Engine AI Text Detection & Humanization Audit Service.

Combines three complementary detection engines on prose chapters:
  1. Engine A (Causal Perplexity & Burstiness — GPTZero / Binoculars style):
     Computes token-level cross-entropy loss, sentence/paragraph Perplexity (PPL),
     and Perplexity Burstiness (sigma_PPL / mu_PPL) using a local GPT-2 causal model.
  2. Engine B (Supervised Neural RoBERTa Classifier):
     Runs a pretrained RoBERTa sequence classifier (`Hello-SimpleAI/chatgpt-detector-roberta`
     with fallback to `openai-community/roberta-base-openai-detector`) over prose windows.
  3. Engine C (Stylometric & Editorial Humanization Auditor):
     Measures sentence-length Burstiness (CV = sigma/mu), Moving-Average Type-Token Ratio
     (MATTR-100), paragraph length variance, and AI-cliche density.

Outputs both a terminal summary and a detailed Markdown audit report (`04_review/ai_detection_audit_report.md`),
and identifies any specific paragraphs that score as overly uniform/synthetic so they can be refined.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

AI_CLICHE_PATTERNS = [
    r"\bdelve(?:s|d)?\s+into\b",
    r"\ba\s+testament\s+to\b",
    r"\brich\s+tapestry\b",
    r"\bin\s+today's\s+fast-paced\b",
    r"\bever-evolving\s+landscape\b",
    r"\bit\s+is\s+important\s+to\s+note\s+that\b",
    r"\bit\s+is\s+worth\s+noting\s+that\b",
    r"\bin\s+conclusion,\b",
    r"\bin\s+summary,\b",
    r"\bmoreover,\b",
    r"\bfurthermore,\b",
    r"\bunlock\s+the\s+full\s+potential\b",
    r"\bparadigm\s+shift\b",
    r"\bgame-changer\b",
    r"\bseamless(?:ly)?\s+integrat",
]


@dataclass
class ChapterDetectionMetrics:
    chapter_file: str
    prose_word_count: int
    paragraph_count: int
    sentence_count: int
    # Engine C: Stylometric & Burstiness
    sent_len_mean: float
    sent_len_std: float
    sent_burstiness_cv: float
    para_len_cv: float
    mattr_100: float
    ai_cliche_count: int
    # Engine A: Causal LM Perplexity & Entropy Burstiness (GPTZero-style)
    mean_perplexity: float
    std_perplexity: float
    perplexity_burstiness: float
    high_entropy_para_ratio: float
    # Engine B: Pretrained RoBERTa Classifier
    roberta_human_prob: float
    roberta_ai_prob: float
    # Composite Humanization Score (0-100, >= 70 = Humanized / Natural Scientific Prose)
    composite_human_score: float
    verdict: str
    flagged_paragraphs: list[dict[str, Any]]


def extract_prose_paragraphs(md_text: str) -> list[str]:
    """Strip fenced code blocks, tables, math blocks, and headings to isolate pure prose paragraphs,
    keeping real inline words intact (only stripping markdown formatting characters).
    """
    # Remove fenced code blocks
    text = re.sub(r"```.*?```", "\n", md_text, flags=re.DOTALL)
    # Remove display math blocks
    text = re.sub(r"\$\$.*?\$\$", "\n", text, flags=re.DOTALL)
    # Strip backticks around inline identifiers while preserving the actual words
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    # Clean inline LaTeX commands to plain readable symbols
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"(?<!\\)\$([^\$\n]+)\$", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    paragraphs: list[str] = []
    for raw_block in re.split(r"\n\s*\n", text):
        lines = [
            ln.strip()
            for ln in raw_block.strip().splitlines()
            if ln.strip()
            and not ln.strip().startswith(("#", "|", "---", "![", "- **Figure", "- **Table"))
        ]
        if not lines:
            continue
        # Skip pure bullet lists with short fragments; keep substantive prose paragraphs (>= 35 words)
        if all(ln.startswith(("-", "*", "1.", "2.", "3.", "4.")) for ln in lines):
            continue
        joined = " ".join(lines)
        joined = re.sub(r"[*_]{1,2}", "", joined)
        words = joined.split()
        if len(words) >= 35:
            paragraphs.append(joined)
    return paragraphs


def compute_mattr(words: list[str], window: int = 100) -> float:
    """Compute Moving-Average Type-Token Ratio (MATTR) over a 100-word sliding window."""
    lower_words = [re.sub(r"[^a-z0-9]", "", w.lower()) for w in words]
    lower_words = [w for w in lower_words if w]
    if len(lower_words) < window:
        return len(set(lower_words)) / max(1, len(lower_words))
    ratios = []
    for i in range(0, len(lower_words) - window + 1, 20):
        chunk = lower_words[i : i + window]
        ratios.append(len(set(chunk)) / window)
    return statistics.mean(ratios) if ratios else 0.0


class LocalAIDetectionService:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        print("[LocalAIDetector] Loading Engine A: Causal Perplexity Model (gpt2)...")
        self.ppl_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.ppl_model = AutoModelForCausalLM.from_pretrained("gpt2").to(self.device)
        self.ppl_model.eval()

        print("[LocalAIDetector] Loading Engine B: Pretrained RoBERTa AI Detector...")
        self.clf_model_name = "Hello-SimpleAI/chatgpt-detector-roberta"
        try:
            self.clf_tokenizer = AutoTokenizer.from_pretrained(self.clf_model_name)
            self.clf_model = AutoModelForSequenceClassification.from_pretrained(
                self.clf_model_name
            ).to(self.device)
        except Exception:
            self.clf_model_name = "openai-community/roberta-base-openai-detector"
            self.clf_tokenizer = AutoTokenizer.from_pretrained(self.clf_model_name)
            self.clf_model = AutoModelForSequenceClassification.from_pretrained(
                self.clf_model_name
            ).to(self.device)
        self.clf_model.eval()
        print(f"[LocalAIDetector] Models loaded successfully on {self.device} ({self.clf_model_name}).")

    @torch.no_grad()
    def compute_perplexity(self, text: str) -> float:
        enc = self.ppl_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        input_ids = enc["input_ids"]
        if input_ids.size(1) < 4:
            return 50.0
        outputs = self.ppl_model(input_ids, labels=input_ids)
        loss = float(outputs.loss.item())
        return min(500.0, math.exp(min(loss, 7.0)))

    @torch.no_grad()
    def predict_ai_probability(self, text: str) -> float:
        """Return probability in [0.0, 1.0] that the text is AI-generated according to RoBERTa."""
        enc = self.clf_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        logits = self.clf_model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()
        # In Hello-SimpleAI/chatgpt-detector-roberta, label 0 = Human, label 1 = ChatGPT
        # In openai-community/roberta-base-openai-detector, label 0 = Fake (AI), label 1 = Real (Human)
        id2label = getattr(self.clf_model.config, "id2label", {})
        label1_name = str(id2label.get(1, "")).lower()
        if "real" in label1_name or "human" in label1_name:
            return float(probs[0])  # index 0 is AI
        return float(probs[1])  # index 1 is AI (ChatGPT)

    def analyze_chapter(self, md_path: Path) -> ChapterDetectionMetrics:
        raw_md = md_path.read_text(encoding="utf-8")
        paragraphs = extract_prose_paragraphs(raw_md)
        full_prose = "\n\n".join(paragraphs)
        all_words = full_prose.split()

        # Sentence segmentation
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", full_prose)
            if len(s.strip().split()) >= 3
        ]
        sent_lengths = [len(s.split()) for s in sentences] if sentences else [15]
        sent_mean = float(statistics.mean(sent_lengths))
        sent_std = float(statistics.pstdev(sent_lengths)) if len(sent_lengths) > 1 else 0.0
        sent_cv = sent_std / max(1.0, sent_mean)

        para_lengths = [len(p.split()) for p in paragraphs] if paragraphs else [60]
        para_cv = (
            float(statistics.pstdev(para_lengths)) / max(1.0, float(statistics.mean(para_lengths)))
            if len(para_lengths) > 1
            else 0.0
        )

        mattr = compute_mattr(all_words, window=100)

        cliche_count = 0
        for pat in AI_CLICHE_PATTERNS:
            cliche_count += len(re.findall(pat, full_prose, flags=re.IGNORECASE))

        # Per-paragraph Perplexity & RoBERTa scoring
        ppls: list[float] = []
        ai_probs: list[float] = []
        flagged: list[dict[str, Any]] = []

        for idx, para in enumerate(paragraphs):
            ppl = self.compute_perplexity(para)
            ai_p = self.predict_ai_probability(para)
            ppls.append(ppl)
            ai_probs.append(ai_p)
            # Flag paragraph if both low perplexity (< 24) and high RoBERTa AI prob (> 0.75)
            if ppl < 24.0 and ai_p > 0.75:
                flagged.append(
                    {
                        "paragraph_index": idx + 1,
                        "perplexity": round(ppl, 2),
                        "roberta_ai_prob": round(ai_p, 3),
                        "excerpt": para[:140] + "...",
                    }
                )

        mean_ppl = float(statistics.mean(ppls)) if ppls else 40.0
        std_ppl = float(statistics.pstdev(ppls)) if len(ppls) > 1 else 10.0
        ppl_burstiness = std_ppl / max(1.0, mean_ppl)
        high_entropy_ratio = (
            sum(1 for x in ppls if x >= 32.0) / max(1, len(ppls))
        )

        mean_ai_prob = float(statistics.mean(ai_probs)) if ai_probs else 0.25
        mean_human_prob = 1.0 - mean_ai_prob

        # Composite Humanization Score (0..100):
        # Combines:
        #   - 35% RoBERTa Human Probability (0..100)
        #   - 30% Causal Perplexity & Entropy Score (PPL >= 35 and PPL Burstiness >= 0.30 -> 100)
        #   - 25% Sentence & Paragraph Rhythm Burstiness (sent_cv >= 0.45 -> 100)
        #   - 10% Lexical Diversity & Zero Cliches (mattr >= 0.70 and 0 cliches -> 100)
        roberta_component = mean_human_prob * 100.0
        ppl_component = min(100.0, (mean_ppl / 42.0) * 65.0 + min(35.0, (ppl_burstiness / 0.35) * 35.0))
        rhythm_component = min(100.0, (sent_cv / 0.48) * 70.0 + min(30.0, (para_cv / 0.35) * 30.0))
        lex_component = max(0.0, min(100.0, (mattr / 0.72) * 100.0 - cliche_count * 8.0))

        composite = (
            0.35 * roberta_component
            + 0.30 * ppl_component
            + 0.25 * rhythm_component
            + 0.10 * lex_component
        )

        if composite >= 72.0:
            verdict = "HUMANIZED_NATURAL_PROSE"
        elif composite >= 58.0:
            verdict = "MIXED_TECHNICAL_PROSE"
        else:
            verdict = "NEEDS_HUMANIZATION_PASS"

        return ChapterDetectionMetrics(
            chapter_file=md_path.name,
            prose_word_count=len(all_words),
            paragraph_count=len(paragraphs),
            sentence_count=len(sentences),
            sent_len_mean=round(sent_mean, 2),
            sent_len_std=round(sent_std, 2),
            sent_burstiness_cv=round(sent_cv, 3),
            para_len_cv=round(para_cv, 3),
            mattr_100=round(mattr, 3),
            ai_cliche_count=cliche_count,
            mean_perplexity=round(mean_ppl, 2),
            std_perplexity=round(std_ppl, 2),
            perplexity_burstiness=round(ppl_burstiness, 3),
            high_entropy_para_ratio=round(high_entropy_ratio, 3),
            roberta_human_prob=round(mean_human_prob, 3),
            roberta_ai_prob=round(mean_ai_prob, 3),
            composite_human_score=round(composite, 1),
            verdict=verdict,
            flagged_paragraphs=flagged,
        )


def main() -> None:
    book_root = Path(__file__).resolve().parent.parent
    manuscript_dir = book_root / "03_manuscript"
    chapters_dir = manuscript_dir / "chapters"
    review_dir = book_root / "04_review"
    review_dir.mkdir(parents=True, exist_ok=True)

    ordered_files = (
        [manuscript_dir / "00_front_matter.md"]
        + sorted(chapters_dir.glob("*.md"))
        + [manuscript_dir / "99_back_matter.md"]
    )

    service = LocalAIDetectionService(device="cpu")
    results: list[ChapterDetectionMetrics] = []

    print("\n=== RUNNING MULTI-ENGINE AI DETECTION AUDIT ACROSS BOOK-002 ===")
    for fpath in ordered_files:
        metrics = service.analyze_chapter(fpath)
        results.append(metrics)
        print(
            f"  - {metrics.chapter_file:<46} | "
            f"PPL={metrics.mean_perplexity:>5.1f} (Burst={metrics.perplexity_burstiness:.2f}) | "
            f"SentCV={metrics.sent_burstiness_cv:.2f} | "
            f"RoBERTa_Human={metrics.roberta_human_prob*100:>5.1f}% | "
            f"Cliches={metrics.ai_cliche_count} | "
            f"Score={metrics.composite_human_score:>5.1f}/100 [{metrics.verdict}]"
        )

    avg_score = statistics.mean(r.composite_human_score for r in results)
    avg_ppl = statistics.mean(r.mean_perplexity for r in results)
    avg_sent_cv = statistics.mean(r.sent_burstiness_cv for r in results)
    avg_roberta_human = statistics.mean(r.roberta_human_prob for r in results)
    total_cliches = sum(r.ai_cliche_count for r in results)
    total_flagged = sum(len(r.flagged_paragraphs) for r in results)

    print("\n=== GLOBAL BOOK-002 HUMANIZATION & AI DETECTION SUMMARY ===")
    print(f"  - Global Composite Humanization Score : {avg_score:.1f} / 100")
    print(f"  - Mean Causal Perplexity (GPT-2)      : {avg_ppl:.2f} (Human threshold >= 32.0)")
    print(f"  - Mean Sentence Burstiness (CV)       : {avg_sent_cv:.3f} (Human threshold >= 0.42)")
    print(f"  - Mean Pretrained RoBERTa Human Prob  : {avg_roberta_human*100:.1f}%")
    print(f"  - Total AI Cliche Matches Across Book : {total_cliches}")
    print(f"  - Total Flagged Uniform Paragraphs    : {total_flagged}")

    # Write detailed report
    report_path = review_dir / "ai_detection_audit_report.md"
    json_path = review_dir / "ai_detection_audit_report.json"
    json_path.write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )

    md_lines = [
        "# Multi-Engine AI Detection & Humanization Audit Report — BOOK-002 (*Why AI Demos Lie*)",
        "",
        "## 1. Executive Summary & Detection Engines",
        f"- **Global Composite Humanization Score:** **{avg_score:.1f} / 100**",
        f"- **Engine A (GPT-2 Causal Perplexity & Entropy Burstiness):** Mean PPL = **{avg_ppl:.2f}** (Human technical baseline $\\ge 32.0$)",
        f"- **Engine B (Pretrained RoBERTa Classifier `{service.clf_model_name}`):** Mean Human Probability = **{avg_roberta_human*100:.1f}%** (AI Probability = **{(1-avg_roberta_human)*100:.1f}%**)",
        f"- **Engine C (Stylometric Burstiness & Cliché Scan):** Sentence Length CV = **{avg_sent_cv:.3f}**, Total AI Clichés = **{total_cliches}**",
        "",
        "## 2. Chapter-by-Chapter Breakdown",
        "",
        "| Chapter File | Prose Words | Mean PPL | PPL Burstiness | Sent CV | MATTR-100 | Clichés | RoBERTa Human % | Composite Score | Verdict |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for r in results:
        md_lines.append(
            f"| `{r.chapter_file}` | {r.prose_word_count} | {r.mean_perplexity:.1f} | "
            f"{r.perplexity_burstiness:.2f} | {r.sent_burstiness_cv:.2f} | {r.mattr_100:.2f} | "
            f"{r.ai_cliche_count} | {r.roberta_human_prob*100:.1f}% | **{r.composite_human_score:.1f}/100** | `{r.verdict}` |"
        )

    md_lines.extend(["", "## 3. Flagged Paragraphs Inspection (Low Perplexity < 24 & High RoBERTa AI > 75%)", ""])
    if total_flagged == 0:
        md_lines.append("Zero paragraphs triggered the combined low-perplexity + high-AI classifier threshold.")
    else:
        for r in results:
            for fp in r.flagged_paragraphs:
                md_lines.append(
                    f"- **`{r.chapter_file}` (Para #{fp['paragraph_index']}):** "
                    f"PPL={fp['perplexity']}, RoBERTa_AI={fp['roberta_ai_prob']*100:.1f}% — *\"{fp['excerpt']}\"*"
                )

    report_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"\nDetailed audit report written to: {report_path}")


if __name__ == "__main__":
    main()

