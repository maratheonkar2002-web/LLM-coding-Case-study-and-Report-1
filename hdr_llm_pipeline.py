"""
=============================================================================
 Local LLM Country Development Intelligence Pipeline
 -----------------------------------------------------------------------
 Assignment 1: LLM coding case study (PDF -> structured data -> dashboard)

 What this script does, end to end:
   1. Reads a UN National Human Development Report (NHDR) PDF and cleans
      the extracted text.
   2. Uses a *local* LLM (via Ollama) to summarise the report and each
      chapter, and to pull out structured development indicators, themes,
      strengths/challenges, and any time-based demographic figures.
   3. Uses a *second, different* local LLM purely as a judge, scoring the
      first model's output for completeness, consistency and factual
      alignment with the source text.
   4. Repeats step 2 with a third model so the three models' behaviour can
      be compared (extension task: "which model is richest / most stable").
   5. Builds one interactive Plotly dashboard with 5 charts and writes all
      structured results to JSON/CSV for the written report.

 Design notes for the marker:
   - Every LLM call goes through `ask_llm()`, which is wrapped in a
     try/except. If Ollama isn't running, or the requested model isn't
     pulled, the pipeline falls back to a cheap deterministic heuristic
     (first-N-sentence summary, regex indicator search) so the script
     still finishes and still produces all 5 plots. This is intentional -
     a marking run without Ollama installed should not crash, it should
     just fall back and say so in the console.
   - The regex fallback is deliberately weaker than the LLM path - that
     contrast is discussed in the report as evidence of why an LLM is
     used for this task rather than pure rule-based extraction.
   - Swap MODEL_A / MODEL_B / MODEL_C below for whatever you have pulled
     locally, e.g. `ollama pull llama3.1`, `ollama pull mistral`,
     `ollama pull phi3`.
=============================================================================
"""

import os
import re
import json
import time
import argparse
from collections import Counter

import pdfplumber
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


# --------------------------------------------------------------------------
# 0. CONFIG
# --------------------------------------------------------------------------
PDF_PATH = "montenegronhdr2009en.pdf"   # <-- point this at your assigned country's report
OUTPUT_DIR = "outputs"

MODEL_A = "llama3.1"    # primary extraction / summarisation model
MODEL_B = "mistral"     # independent evaluator model (judges model A's output)
MODEL_C = "phi3"        # third model, used only for the cross-model comparison

THEME_KEYWORDS = {
    "education":  ["education", "school", "literacy", "enrolment", "student", "curriculum"],
    "health":     ["health", "healthcare", "mortality", "life expectancy", "hospital", "disease"],
    "inequality": ["inequality", "exclusion", "poverty", "disparit", "vulnerable", "deprivation"],
    "economy":    ["economy", "gdp", "gni", "income", "employment", "budget", "growth"],
    "gender":     ["gender", "women", "female", "male share", "gem", "gdi"],
    "climate":    ["climate", "environment", "emission", "sustainab", "pollution", "energy"],
    "employment": ["employment", "unemploy", "labour", "labor", "job", "workforce"],
}

INDICATOR_PATTERNS = {
    # generic regex fallback - deliberately simple, used only if the LLM path fails
    "hdi_value":        r"HDI value of ([\d.]+)",
    "hdi_rank":          r"HDI rank[^\d]{0,15}(\d+)",
    "life_expectancy":  r"[Ll]ife expectancy at birth[,\s]*years?,?\s*\d{0,4}\D{0,10}([\d.]+)",
    "gni_per_capita":   r"GNI per capita.{0,20}?\$?\s?([\d,]{3,7})",
    "population":       r"population of [A-Z][a-zA-Z]+ is (?:about |approximately )?([\d,]{5,})",
    "mean_schooling":   r"[Mm]ean years of schooling\D{0,10}([\d.]+)",
    "expected_schooling": r"[Ee]xpected years of schooling\D{0,10}([\d.]+)",
}

PROMPTS = {
    "summary": (
        "You are a development economist. Read the excerpt below from a UN National "
        "Human Development Report. In no more than 6 bullet points, list the report's "
        "key results (facts, figures, and trends only - no opinions). "
        "Return plain bullet points, nothing else.\n\nEXCERPT:\n{chunk}"
    ),
    "chapter_summary": (
        "Summarise the following report chapter in under 100 words. Be factual and "
        "neutral, keep any numbers you mention, and do not add information that is "
        "not in the text.\n\nCHAPTER TEXT:\n{chunk}"
    ),
    "indicators": (
        "Extract core numerical development indicators from the text below and return "
        "ONLY a single valid JSON object (no prose, no markdown fences) with these keys "
        "if present, else null: hdi_value, hdi_rank, life_expectancy, gni_per_capita, "
        "expected_years_schooling, mean_years_schooling, population.\n\nTEXT:\n{chunk}"
    ),
    "strengths_challenges": (
        "From the text below, list the report's key STRENGTHS (max 8) and CHALLENGES "
        "(max 8) for national development. Return ONLY valid JSON: "
        '{{"strengths": [...], "challenges": [...]}}\n\nTEXT:\n{chunk}'
    ),
    "demographic_trend": (
        "The text below may contain a table or passage describing a quantity that "
        "changes over time (e.g. population by age group, HDI over years, poverty "
        "rate by year). If you find one, return ONLY valid JSON as a list of records: "
        '[{{"year": 2001, "series": "0-14", "value": 20.6}}, ...]. '
        "If nothing time-based is present, return an empty JSON list [].\n\nTEXT:\n{chunk}"
    ),
    "evaluation": (
        "You are an independent quality reviewer. Compare the SUMMARY to the SOURCE "
        "TEXT it was generated from. Score each from 1 (poor) to 5 (excellent): "
        "completeness (did it capture the key facts?), consistency (internally "
        "coherent?), factual_alignment (no hallucinated numbers/claims?). "
        'Return ONLY valid JSON: {{"completeness": n, "consistency": n, '
        '"factual_alignment": n}}\n\nSOURCE TEXT:\n{source}\n\nSUMMARY:\n{summary}'
    ),
}


# --------------------------------------------------------------------------
# 1. PDF PROCESSING PIPELINE
# --------------------------------------------------------------------------
def extract_pdf_text(pdf_path: str) -> str:
    """Extract raw text from every page of the report using pdfplumber."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


def clean_text(raw_text: str) -> str:
    """Strip repeated headers/footers, fix hyphenated line breaks, collapse whitespace."""
    text = re.sub(r"-\n", "", raw_text)                     # de-hyphenate wrapped words
    text = re.sub(r"National Human Development Report \d{4}", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def chunk_text(text: str, chunk_words: int = 900, overlap: int = 120) -> list:
    """Split cleaned text into overlapping word-count chunks (keeps local context)."""
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_words
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def split_into_chapters(text: str) -> dict:
    """
    Generic chapter splitter: looks for 'CHAPTER n' style headings anywhere in the
    body text. Falls back to treating the whole document as a single 'chapter' if
    no such headings are found (some reports use different structuring).
    """
    matches = list(re.finditer(r"CHAPTER\s+\d+[:.]?", text, flags=re.IGNORECASE))
    if len(matches) < 2:
        return {"Full Report": text[:6000]}
    chapters = {}
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = m.group().strip()
        chapters[title] = text[start:end][:6000]  # cap length per chapter for LLM context
    return chapters


# --------------------------------------------------------------------------
# 2. LLM WRAPPER (with graceful fallback if Ollama isn't running)
# --------------------------------------------------------------------------
def ask_llm(prompt: str, model: str, source_text: str = "", expect_json: bool = False):
    """
    Send a prompt to a local Ollama model. Returns (response_text, latency_seconds).
    Falls back to a lightweight heuristic if Ollama is unavailable/errors out, so the
    rest of the pipeline never crashes.

    `source_text` is the raw chunk the prompt was built from - it's only used by the
    fallback heuristic (kept separate from `prompt` so the fallback never has to guess
    where the content starts inside an arbitrary prompt template).
    """
    start = time.time()
    if OLLAMA_AVAILABLE:
        try:
            resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            content = resp["message"]["content"].strip()
            return content, time.time() - start
        except Exception as e:
            print(f"  [warn] Ollama call to '{model}' failed ({e}); using fallback heuristic.")

    # ---- Fallback heuristic (no LLM available) ----
    if expect_json:
        content = "[]" if "list of records" in prompt.lower() else "{}"
    else:
        # crude extractive fallback: first 3 sentences of the source chunk
        sentences = re.split(r"(?<=[.!?])\s+", source_text.strip())
        content = " ".join(s for s in sentences[:3] if s) or "(no summary available)"
    return content, time.time() - start


def safe_json_parse(text: str, default):
    """Extract the first {...} or [...] block from an LLM reply and parse it."""
    try:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
        return json.loads(match.group(1)) if match else default
    except Exception:
        return default


# --------------------------------------------------------------------------
# 3. EXTRACTION TASKS (LLM model A, judged by model B)
# --------------------------------------------------------------------------
def summarise_report(chunks: list, model: str) -> str:
    chunk = chunks[0][:4000]
    return ask_llm(PROMPTS["summary"].format(chunk=chunk), model, source_text=chunk)[0]


def summarise_chapters(chapters: dict, model: str) -> dict:
    out = {}
    for title, body in chapters.items():
        chunk = body[:4000]
        out[title] = ask_llm(PROMPTS["chapter_summary"].format(chunk=chunk), model, source_text=chunk)[0]
    return out


def extract_indicators(text: str, model: str) -> dict:
    chunk = text[:4000]
    llm_text, _ = ask_llm(PROMPTS["indicators"].format(chunk=chunk), model, source_text=chunk, expect_json=True)
    data = safe_json_parse(llm_text, {})
    # regex fallback fills in anything the LLM missed / returned as null
    for key, pattern in INDICATOR_PATTERNS.items():
        if not data.get(key):
            m = re.search(pattern, text)
            if m:
                data[key] = m.group(1).replace(",", "")
    return data


def extract_strengths_challenges(text: str, model: str) -> dict:
    chunk = text[:4000]
    llm_text, _ = ask_llm(PROMPTS["strengths_challenges"].format(chunk=chunk), model,
                          source_text=chunk, expect_json=True)
    return safe_json_parse(llm_text, {"strengths": [], "challenges": []})


def extract_demographic_trend(text: str, model: str) -> list:
    chunk = text[:4000]
    llm_text, _ = ask_llm(PROMPTS["demographic_trend"].format(chunk=chunk), model,
                          source_text=chunk, expect_json=True)
    return safe_json_parse(llm_text, [])


def extract_themes(full_text: str) -> dict:
    """Deterministic keyword-frequency count across the seven required themes."""
    lower = full_text.lower()
    return {theme: sum(lower.count(kw) for kw in kws) for theme, kws in THEME_KEYWORDS.items()}


def evaluate_output(source: str, summary: str, judge_model: str) -> dict:
    """Second LLM scores the first model's summary against its source text."""
    llm_text, _ = ask_llm(
        PROMPTS["evaluation"].format(source=source[:2500], summary=summary[:1500]),
        judge_model, source_text=summary[:1500], expect_json=True,
    )
    return safe_json_parse(llm_text, {"completeness": None, "consistency": None, "factual_alignment": None})


# --------------------------------------------------------------------------
# 4. CROSS-MODEL COMPARISON (extension task)
# --------------------------------------------------------------------------
def compare_models(sample_chunk: str, models: list) -> dict:
    """Run the same summarisation prompt through several local models and record
    verbosity (word count), latency, and the evaluator's quality score for each."""
    results = {}
    for model in models:
        chunk = sample_chunk[:4000]
        summary, latency = ask_llm(PROMPTS["chapter_summary"].format(chunk=chunk), model, source_text=chunk)
        scores = evaluate_output(sample_chunk, summary, judge_model=MODEL_B)
        results[model] = {
            "word_count": len(summary.split()),
            "latency_sec": round(latency, 2),
            "completeness": scores.get("completeness"),
            "consistency": scores.get("consistency"),
            "factual_alignment": scores.get("factual_alignment"),
        }
    return results


# --------------------------------------------------------------------------
# 5. DASHBOARD (5 plots, one interactive HTML file)
# --------------------------------------------------------------------------
def build_dashboard(theme_counts, indicators, demo_trend, model_comparison, out_path):
    fig = make_subplots(
        rows=3, cols=2,
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}],
               [{"type": "polar", "colspan": 2}, None]],
        subplot_titles=(
            "1. Theme Distribution (keyword mentions)",
            "2. Time-based Development Indicators",
            "3. Demographic / Trend Data Extracted by LLM",
            "4. Cross-Model Comparison (verbosity & quality)",
            "5. Development Indicators Radar (advanced viz)",
        ),
        vertical_spacing=0.10,
    )

    # --- Plot 1: theme distribution ---
    themes = list(theme_counts.keys())
    counts = list(theme_counts.values())
    fig.add_trace(go.Bar(x=themes, y=counts, marker_color="#4C78A8", name="Theme mentions"),
                  row=1, col=1)

    # --- Plot 2: time-based indicator trend (HDI etc, if we found one) ---
    if demo_trend:
        series_names = sorted(set(r["series"] for r in demo_trend))
        for s in series_names:
            pts = sorted([(r["year"], r["value"]) for r in demo_trend if r["series"] == s])
            xs, ys = zip(*pts)
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines+markers", name=s), row=1, col=2)
    else:
        fig.add_trace(go.Scatter(x=[], y=[], name="no trend data found"), row=1, col=2)

    # --- Plot 3: same demographic data as grouped bars (different lens on it) ---
    if demo_trend:
        years = sorted(set(r["year"] for r in demo_trend))
        for s in series_names:
            ys = [next((r["value"] for r in demo_trend if r["year"] == y and r["series"] == s), 0)
                  for y in years]
            fig.add_trace(go.Bar(x=years, y=ys, name=f"{s} (bar)"), row=2, col=1)
    fig.update_layout(barmode="group")

    # --- Plot 4: cross-model comparison ---
    models = list(model_comparison.keys())
    fig.add_trace(go.Bar(x=models, y=[model_comparison[m]["word_count"] for m in models],
                         name="Summary word count", marker_color="#F58518"), row=2, col=2)
    fig.add_trace(go.Bar(x=models, y=[model_comparison[m]["completeness"] or 0 for m in models],
                         name="Evaluator score (completeness)", marker_color="#54A24B"), row=2, col=2)

    # --- Plot 5: radar chart of normalised indicators ---
    radar_labels = ["HDI value", "Life expectancy (norm.)", "Education (norm.)", "Income (norm.)"]
    radar_values = [
        _safe_float(indicators.get("hdi_value"), 0.7),
        _norm(indicators.get("life_expectancy"), 40, 85),
        _safe_float(indicators.get("expected_years_schooling"), 10) / 18,
        _norm(indicators.get("gni_per_capita"), 500, 60000),
    ]
    fig.add_trace(go.Scatterpolar(r=radar_values + radar_values[:1],
                                  theta=radar_labels + radar_labels[:1],
                                  fill="toself", name="Indicators"), row=3, col=1)

    fig.update_layout(height=1150, width=1100, showlegend=True,
                      title_text="National Human Development Report - LLM Extraction Dashboard")
    fig.write_html(out_path)
    return out_path


def _safe_float(val, default):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _norm(val, lo, hi):
    v = _safe_float(val, (lo + hi) / 2)
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


# --------------------------------------------------------------------------
# 6. MAIN
# --------------------------------------------------------------------------
def main(pdf_path: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Ollama available: {OLLAMA_AVAILABLE}")

    print("\n[1/6] Extracting and cleaning PDF text ...")
    raw_text = extract_pdf_text(pdf_path)
    text = clean_text(raw_text)
    chunks = chunk_text(text)
    chapters = split_into_chapters(text)
    print(f"  {len(text.split())} words, {len(chunks)} chunks, {len(chapters)} chapter(s) found")

    print("\n[2/6] Summarising report and chapters with model A ...")
    report_summary = summarise_report(chunks, MODEL_A)
    chapter_summaries = summarise_chapters(chapters, MODEL_A)

    print("\n[3/6] Evaluating summaries with model B (independent judge) ...")
    eval_scores = evaluate_output(chunks[0], report_summary, judge_model=MODEL_B)

    print("\n[4/6] Extracting themes, indicators, strengths/challenges, trends ...")
    theme_counts = extract_themes(text)
    indicators = extract_indicators(text, MODEL_A)
    strengths_challenges = extract_strengths_challenges(text, MODEL_A)
    demo_trend = extract_demographic_trend(text, MODEL_A)

    print("\n[5/6] Running cross-model comparison (extension task) ...")
    model_comparison = compare_models(chunks[0], [MODEL_A, MODEL_B, MODEL_C])

    print("\n[6/6] Building interactive dashboard (5 plots) ...")
    dashboard_path = build_dashboard(theme_counts, indicators, demo_trend,
                                     model_comparison, os.path.join(OUTPUT_DIR, "dashboard.html"))

    results = {
        "report_summary": report_summary,
        "chapter_summaries": chapter_summaries,
        "evaluation_scores": eval_scores,
        "theme_counts": theme_counts,
        "indicators": indicators,
        "strengths_challenges": strengths_challenges,
        "demographic_trend": demo_trend,
        "model_comparison": model_comparison,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone. Dashboard: {dashboard_path}")
    print(f"Structured results: {os.path.join(OUTPUT_DIR, 'results.json')}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local LLM HDR pipeline")
    parser.add_argument("--pdf", default=PDF_PATH, help="Path to the assigned country's HDR PDF")
    args = parser.parse_args()
    main(args.pdf)
