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

---

## One-Page Writeup

### 1. Approach to the Reasoning Problem

The core challenge is **temporal causal inference from noisy, unstructured health logs** — not keyword matching. My approach:

**A. Context-rich encoding**: Every session is enriched with a computed week number before being sent to the LLM. This transforms abstract session IDs into calendar-anchored events the model can reason about (e.g., "Week 2 diet change → Week 10 hair fall").

**B. Instruction-forced reasoning chain**: The system prompt explicitly forbids keyword association and requires: (1) time-gap calculation, (2) physiological mechanism explanation, (3) dose-response or cycle detection, (4) distinction between cause and correlation. The model must show its work.

**C. Differential diagnosis via confidence scoring**: Each pattern must include a `counter_argument` field — forcing the model to consider why it might be wrong before committing to a confidence level.

**D. "What Clary missed" framing**: The prompt instructs the model to find patterns *not already surfaced* in Clary's past responses. This prevents the system from simply rewrapping existing diagnoses.

**Patterns targeted (8 planted)**:
1. Arjun: Late eating → stomach pain (temporal cluster on deadline nights) ✅
2. Arjun: Low water + busy days → afternoon headaches (repeating cycle) ✅
3. Arjun: Sedentary work → lower back pain ✅
4. Meera: Calorie restriction (Jan) → hair fall (Feb, 6-7 week lag) ✅
5. Meera: High dairy → cheek/jawline acne (dose-response) ✅
6. Meera: Under-fuelling → workout fatigue + brain fog ✅
7. Priya: High-carb lunch → post-lunch energy crash ✅
8. Priya: Late-night screens → sleep disruption → worsened period cramps (cortisol chain) ✅

### 2. Where the System Fails or Hallucinates Confidently

**Failure Mode 1: Temporal precision hallucination**  
The model is told "hair fall manifests 8-12 weeks after deficiency" and will confidently apply this to the dataset even if the actual gap is 6 weeks. It knows the medical rule but applies it loosely. **Mitigation**: Explicitly compute day-gaps in preprocessing and pass exact numbers rather than letting the LLM estimate.

**Failure Mode 2: Confounding variable blindness**  
The model might attribute Priya's worsened cramps entirely to screens/sleep without adequately weighting stress as a co-variable. When two variables change simultaneously, the model tends to pick the most recent or most "interesting" one.

**Failure Mode 3: Session sampling bias**  
If a user mentions a symptom once and a trigger 3 times, the LLM weights the trigger as causally strong. Frequency of mention ≠ strength of causation — but LLMs tend to conflate the two.

**Failure Mode 4: JSON extraction brittleness**  
The `<patterns>` tag extraction can fail on malformed JSON (trailing commas, unescaped quotes in strings). Current fallback regex helps but doesn't fully recover. **Fix**: Use Anthropic's tool-calling / structured output mode for guaranteed JSON schema compliance.

### 3. What I'd Build Differently With More Time

**A. Preprocessing layer**: Compute all temporal gaps, session deltas, and symptom timelines *before* the LLM call. Pass pre-calculated metadata so the model reasons about numbers, not estimates them.

**B. Multi-pass reasoning**: Pass 1 = extract all events into a structured timeline. Pass 2 = hypothesis generation. Pass 3 = hypothesis validation against timeline. Reduces hallucination per pass.

**C. Retrieval-augmented grounding**: For medical claims (e.g., telogen effluvium lag times), retrieve from a curated medical knowledge base rather than relying on training data, which may be stale or wrong.

**D. Contradiction detection**: A second LLM call that reads the detected patterns and actively tries to disprove each one using the same conversation history.

**E. Longitudinal drift detection**: Track how patterns evolve across re-runs as new sessions are added — not just a static analysis but a living health graph.
