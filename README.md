# Clary · Ask First Health Pattern Intelligence

A Streamlit app that detects hidden cross-conversation health patterns using temporal reasoning, powered by Claude Sonnet 4.

---

## File Structure

```
ask-first/
├── app.py                          # Main Streamlit application
├── askfirst_synthetic_dataset.json # Input dataset
├── requirements.txt                # google-genai, streamlit
└── README.md
```

---

## Setup & Run

### 1 · Clone the repo

```bash
git clone https://github.com/shrutisingh004/ask-first.git
cd ask-first
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

### 3 · Set your API key

Create a `.env` file in the root:

```env
GOOGLE_API_KEY = your_gemini_api_key_here
```

> Get a free Gemini API key at [aistudio.google.com](https://aistudio.google.com)

### 4 · Run the app

```bash
streamlit run app.py
```
Then open `http://localhost:8501`, select a user, and click **Analyze Patterns**.

---

## Model Choice: Gemini 2.5 Flash
 
**Why Gemini 2.5 Flash?**
- **Completely free** via Google AI Studio — no credit card, no billing, 15 requests/minute on the free tier.
- **1M token context window** — far larger than competing free models (Groq's Llama 3 tops out at 8K–32K). As Ask First scales to longer user histories, Gemini stays safe.
- **Built-in thinking mode** — Gemini 2.5 Flash has an internal reasoning step before responding, which significantly improves multi-hop temporal causal inference (the core challenge here) compared to 1.5 Flash.
- **Stronger reasoning overall** — 2.5 Flash outperforms 1.5 Flash on complex reasoning benchmarks, making it better suited for connecting health events across weeks of messy, unstructured conversation data.
- **Reliable JSON output** — follows complex output format instructions (the `<patterns>` tag contract) consistently.
- **Native streaming** via `client.models.generate_content_stream()` — enables real-time reasoning trace visibility in the UI.

---

## Architecture & Context Management Strategy

### Chunking Strategy
**Full-history, single-pass**: Unlike RAG approaches that chunk conversations into separate embeddings, this system feeds the **entire user history in one context window** (~2000–3000 tokens per user). 

**Why?** Pattern detection requires global temporal awareness. A hair-fall symptom in Week 11 only becomes meaningful when the system can "see" the diet change in Week 1. Chunked retrieval would break these long-range dependencies.

**What's included per call:**
- Full user profile (age, occupation, lifestyle notes)
- All sessions in chronological order with: timestamp, week number, user message, followup, tags, severity
- Clary's past responses (to know what was already surfaced vs. what's hidden)

**Token budget**: ~2500 tokens of history + 500 token system prompt + 4000 token completion budget = well within 200k context.

### Reasoning Trace Strategy
The system prompt forces the LLM to output a `<reasoning_trace>` block **before** the JSON. This is a chain-of-thought forcing function — the model reasons aloud, considers alternative explanations, calculates time gaps, then commits to pattern conclusions. The trace is surfaced in the UI so evaluators can audit the reasoning.

### Temporal Encoding
Each session is explicitly tagged with:
- ISO timestamp  
- Human-readable week number (Week N from Jan 1, 2026)
- This primes the model to reason in calendar weeks, not just session numbers.
