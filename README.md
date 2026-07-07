# Local LLM Country Development Intelligence Pipeline

Turns a UN National Human Development Report (NHDR) PDF into structured
data and an interactive 5-plot dashboard, using local LLMs for extraction,
summarisation, and evaluation.

## 1. Setup

```bash
pip install -r requirements.txt

# Optional but recommended - install Ollama and pull 3 small local models
# https://ollama.com
ollama pull llama3.1   # extraction / summarisation
ollama pull mistral    # independent evaluator
ollama pull phi3       # third model, used for the cross-model comparison
```

If Ollama isn't installed or running, the script still runs end-to-end -
it falls back to a simple extractive heuristic and regex indicator search
so you always get a finished dashboard. This is intentional (see the
"Design notes" comment block at the top of `hdr_llm_pipeline.py`) and is
worth mentioning in your report as a real comparison point between
LLM-based and rule-based extraction.

## 2. Run it

```bash
python hdr_llm_pipeline.py --pdf your_country_report.pdf
```

Replace `your_country_report.pdf` with whichever country's NHDR you were
assigned (check the module's supplementary files page). By default it
points at `montenegronhdr2009en.pdf`.

## 3. Outputs

- `outputs/dashboard.html` - interactive Plotly dashboard, 5 plots:
  1. Theme distribution (keyword mentions across education, health,
     inequality, economy, gender, climate, employment)
  2. Time-based development indicators (line chart)
  3. The same time-based data as grouped bars (a second lens on it)
  4. Cross-model comparison (verbosity + evaluator score per model)
  5. Radar chart of normalised development indicators (advanced viz)
- `outputs/results.json` - all structured results (summaries, indicators,
  theme counts, strengths/challenges, demographic trend, model comparison)
  in machine-readable form.

## 4. How this maps to the assignment brief

| Brief requirement | Where it's handled |
|---|---|
| PDF → text → chunks | `extract_pdf_text`, `clean_text`, `chunk_text` |
| Chapter summaries (<100 words) | `summarise_chapters` (model A) |
| Full-report key results | `summarise_report` (model A) |
| Second LLM evaluates first LLM's output | `evaluate_output` (model B) |
| Theme distribution counts | `extract_themes` |
| Strengths / challenges | `extract_strengths_challenges` |
| Structured numeric indicators (JSON) | `extract_indicators` |
| Demographic/time-based trend | `extract_demographic_trend` |
| Interactive dashboard, ≥4 plots | `build_dashboard` (5 plots) |
| Extension: 3-model comparison | `compare_models` |
| Extension: advanced plot | Radar chart (plot 5) |

The optimised prompts used for every step are the `PROMPTS` dict near the
top of the script - copy these into your report's "prompts used" section.
