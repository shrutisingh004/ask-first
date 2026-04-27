import os
import streamlit as st
import json
import time
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Config
st.set_page_config(
    page_title="Clary · Ask First",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = Path(__file__).parent / "askfirst_synthetic_dataset.json"

# Load data
@st.cache_data
def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)

def format_user_history(user: dict) -> str:
    """Format a user's full conversation history as a rich text block for the LLM."""
    lines = []
    lines.append(f"USER PROFILE")
    lines.append(f"Name: {user['name']}, Age: {user['age']}, Gender: {user['gender']}")
    lines.append(f"Occupation: {user['occupation']}, Location: {user['location']}")
    lines.append(f"Onboarding notes: {user['onboarding_notes']}")
    lines.append("")
    lines.append("CONVERSATION HISTORY (chronological):")
    for i, conv in enumerate(user["conversations"], 1):
        ts = conv["timestamp"]
        dt = datetime.fromisoformat(ts)
        week_num = ((dt - datetime(2026, 1, 1)).days // 7) + 1
        lines.append(f"\n--- Session {i} | {dt.strftime('%b %d, %Y')} | ~Week {week_num} ---")
        lines.append(f"User: {conv['user_message']}")
        if conv.get("user_followup"):
            lines.append(f"User followup: {conv['user_followup']}")
        lines.append(f"Clary response: {conv['clary_response']}")
        lines.append(f"Severity: {conv.get('severity','unknown')} | Tags: {', '.join(conv.get('tags',[]))}")
    return "\n".join(lines)

# Prompts
PATTERN_DETECTION_SYSTEM = """You are Clary, an advanced health pattern reasoning engine for the Ask First platform.
Your job is to analyze a user's full health conversation history spanning multiple months and surface HIDDEN PATTERNS that the user has NOT connected themselves.

REASONING RULES:
1. TEMPORAL REASONING IS MANDATORY. A symptom appearing 8-12 weeks after a lifestyle change is causally different from a symptom appearing simultaneously. Always calculate the gap in days/weeks between cause and effect.
2. Find patterns across sessions, not within them. Single-session observations are NOT patterns.
3. Prioritize causal chains: lifestyle change → physiological response → symptom. Show the chain.
4. Consider delayed-onset effects (e.g., nutritional deficiency causing hair fall 8-12 weeks later; sleep debt causing hormonal changes over weeks).
5. Look for: dose-response relationships, temporal clustering, symptom cycles, trigger-symptom pairs.
6. DO NOT surface patterns already explicitly stated in Clary's past responses — find what Clary MISSED.
7. Show your reasoning trace before each conclusion.

OUTPUT FORMAT: For each pattern, output a JSON object with these exact fields:
{
  "pattern_id": "P1",
  "title": "Short descriptive title",
  "sessions_involved": ["S01", "S04", "S07"],
  "trigger": "What causes or precedes the symptom",
  "symptom": "What manifests",
  "temporal_gap_days": 0,
  "temporal_reasoning": "Detailed explanation of the time relationship",
  "reasoning_trace": "Step-by-step reasoning showing what you considered",
  "evidence_strength": "strong|moderate|weak",
  "confidence": "high|medium|low",
  "confidence_justification": "One sentence explaining why this is real, not coincidental",
  "recommendation": "Specific, actionable recommendation"
}

Output a JSON array of ALL patterns found. Be exhaustive — find every cross-session pattern, not just the obvious ones.
Start with a <reasoning_trace> block showing your analysis, then output the JSON array inside <patterns> tags."""

def build_pattern_prompt(user_history: str) -> str:
    return f"""Analyze this user's complete health history and find ALL cross-session patterns with temporal reasoning.

{user_history}

Think step by step. First explore all possible connections. Calculate exact time gaps. Then output findings."""

CONFIDENCE_SYSTEM = """You are a medical reasoning validator. Given a list of health patterns detected from conversation data, evaluate each one's confidence score and explain WHY it's a real connection versus coincidence.

For each pattern, output ONLY a JSON array with:
{
  "pattern_id": "P1", 
  "confidence": "high|medium|low",
  "justification": "One sentence: what makes this causal not coincidental",
  "counter_argument": "What could make this wrong",
  "overall_score": 0-100
}"""

# Gemini client
def get_client():
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)

def parse_patterns_from_response(response_text: str) -> list:
    """Extract JSON patterns from LLM response."""
    # Try to find JSON array inside <patterns> tags
    patterns_match = re.search(r'<patterns>(.*?)</patterns>', response_text, re.DOTALL)
    if patterns_match:
        json_str = patterns_match.group(1).strip()
    else:
        # Try to find raw JSON array
        json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            return []
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common JSON issues
        try:
            json_str = re.sub(r',\s*}', '}', json_str)
            json_str = re.sub(r',\s*]', ']', json_str)
            return json.loads(json_str)
        except:
            return []

def confidence_color(conf: str) -> str:
    return {"high": "#22c55e", "medium": "#f59e0b", "low": "#ef4444"}.get(conf, "#94a3b8")

def evidence_label(strength: str) -> str:
    return {"strong": "[STRONG]", "moderate": "[MODERATE]", "weak": "[WEAK]"}.get(strength, "[?]")

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #0a0d14;
    --surface: #111520;
    --surface2: #161b2e;
    --border: #1e2a45;
    --accent: #4f8ef7;
    --accent2: #7c3aed;
    --text: #e2e8f0;
    --muted: #64748b;
    --green: #22c55e;
    --yellow: #f59e0b;
    --red: #ef4444;
}

.stApp {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

/* Main header */
.clary-header {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 32px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
}
.clary-logo {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
}
.clary-title {
    font-family: 'DM Serif Display', serif;
    font-size: 32px;
    color: var(--text);
    margin: 0;
}
.clary-subtitle {
    color: var(--muted);
    font-size: 14px;
    margin: 2px 0 0;
    font-weight: 300;
}

/* Pattern cards */
.pattern-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.pattern-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.pattern-title {
    font-family: 'DM Serif Display', serif;
    font-size: 20px;
    color: var(--text);
    margin-bottom: 8px;
}
.pattern-meta {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}
.badge {
    padding: 4px 12px;
    border-radius: 100px;
    font-size: 12px;
    font-weight: 500;
    font-family: 'JetBrains Mono', monospace;
}
.badge-sessions {
    background: rgba(79,142,247,0.15);
    color: var(--accent);
    border: 1px solid rgba(79,142,247,0.3);
}
.badge-confidence {
    border: 1px solid currentColor;
}
.trace-block {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #94a3b8;
    white-space: pre-wrap;
    margin-top: 12px;
}
.cause-effect {
    display: flex;
    align-items: center;
    gap: 12px;
    background: var(--surface2);
    border-radius: 12px;
    padding: 16px;
    margin: 12px 0;
    flex-wrap: wrap;
}
.cause-box {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 8px;
    padding: 8px 16px;
    color: #fca5a5;
    font-size: 14px;
}
.effect-box {
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 8px;
    padding: 8px 16px;
    color: #86efac;
    font-size: 14px;
}
.arrow-box {
    color: var(--muted);
    font-size: 20px;
}
.temporal-chip {
    background: rgba(124,58,237,0.15);
    border: 1px solid rgba(124,58,237,0.3);
    border-radius: 8px;
    padding: 8px 16px;
    color: #c4b5fd;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
}
.rec-box {
    background: rgba(79,142,247,0.08);
    border: 1px solid rgba(79,142,247,0.2);
    border-radius: 8px;
    padding: 12px 16px;
    color: #93c5fd;
    font-size: 14px;
    margin-top: 12px;
}
.json-output {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #7ee787;
    overflow-x: auto;
    white-space: pre;
    max-height: 500px;
    overflow-y: auto;
}
.stream-container {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    color: #94a3b8;
    white-space: pre-wrap;
    max-height: 400px;
    overflow-y: auto;
}
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.stat-number {
    font-family: 'DM Serif Display', serif;
    font-size: 36px;
    color: var(--accent);
}
.stat-label {
    color: var(--muted);
    font-size: 13px;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px; font-family: DM Serif Display, serif; font-size: 22px; color: #e2e8f0;'>
        Clary
    </div>
    <div style='color: #64748b; font-size: 12px; margin-bottom: 24px;'>Ask First · Health Reasoning Engine</div>
    """, unsafe_allow_html=True)

    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        st.success("GEMINI_API_KEY loaded")
    else:
        st.error("GEMINI_API_KEY not found in .env")

    st.divider()

    data = load_data()
    users = data["users"]
    user_names = [u["name"] for u in users]

    st.markdown("**Select User**")
    selected_user_name = st.radio(
        "User",
        user_names,
        label_visibility="collapsed"
    )
    selected_user = next(u for u in users if u["name"] == selected_user_name)

    st.divider()
    st.markdown(f"""
    <div style='font-size:12px; color: #64748b;'>
    <b style='color:#94a3b8'>{selected_user['name']}</b><br>
    Age {selected_user['age']} · {selected_user['gender']}<br>
    {selected_user['occupation']}<br>
    {selected_user['location']}<br><br>
    {selected_user['onboarding_notes']}
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    show_raw = st.checkbox("Show raw LLM stream", value=False)
    show_json = st.checkbox("Show JSON output", value=True)

# Main content
st.markdown("""
<div class='clary-header'>
    <div class='clary-logo'>C</div>
    <div>
        <div class='clary-title'>Clary · Pattern Intelligence</div>
        <div class='clary-subtitle'>Cross-conversation health pattern detection with temporal reasoning</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Session timeline
with st.expander(f"{selected_user['name']}'s Conversation Timeline ({len(selected_user['conversations'])} sessions)", expanded=False):
    for i, conv in enumerate(selected_user['conversations'], 1):
        dt = datetime.fromisoformat(conv['timestamp'])
        week_num = ((dt - datetime(2026, 1, 1)).days // 7) + 1
        sev_color = {"mild": "#f59e0b", "moderate": "#ef4444", "none": "#22c55e"}.get(conv.get("severity",""), "#94a3b8")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            st.markdown(f"""
            <div style='text-align:center; padding: 8px;'>
                <div style='font-family: JetBrains Mono; font-size: 10px; color: #64748b;'>S{i:02d}</div>
                <div style='font-size: 11px; color: #94a3b8;'>{dt.strftime('%b %d')}</div>
                <div style='font-size: 10px; color: #64748b;'>Wk {week_num}</div>
                <div style='width:8px; height:8px; border-radius:50%; background:{sev_color}; margin: 4px auto;'></div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            tags_html = " ".join([f"<span style='background: rgba(79,142,247,0.1); color: #93c5fd; font-size: 10px; padding: 2px 8px; border-radius: 100px; border: 1px solid rgba(79,142,247,0.2);'>{t}</span>" for t in conv.get('tags',[])])
            st.markdown(f"""
            <div style='padding: 8px 0; border-bottom: 1px solid #1e2a45;'>
                <div style='font-size: 14px; color: #e2e8f0; margin-bottom: 4px;'>{conv['user_message'][:120]}{'...' if len(conv['user_message'])>120 else ''}</div>
                <div style='margin-top: 4px;'>{tags_html}</div>
            </div>
            """, unsafe_allow_html=True)

# Analysis trigger
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    analyze_btn = st.button(
        f"Analyze {selected_user['name']}'s Patterns",
        type="primary",
        use_container_width=True,
        disabled=not os.getenv("GEMINI_API_KEY")
    )

if not os.getenv("GEMINI_API_KEY"):
    st.info("`GEMINI_API_KEY` not found. Add it to your `.env` file and restart the app.")

# Analysis results
if analyze_btn:
    client = get_client()
    if not client:
        st.error("No API key set.")
    else:
        st.markdown(f"### Reasoning for {selected_user['name']}")
        
        raw_placeholder = st.empty()
        status = st.status("Running temporal pattern analysis...", expanded=True)
        
        full_response = ""
        raw_chunks = []
        
        with status:
            st.write("Loading full conversation history...")
            time.sleep(0.3)
            st.write("Building cross-session context window...")
            time.sleep(0.3)
            st.write("Streaming LLM reasoning...")
            
            # Stream the response
            stream_box = st.empty()
            
            try:
                history = format_user_history(selected_user)
                prompt = build_pattern_prompt(history)
                
                full_prompt = f"{PATTERN_DETECTION_SYSTEM}\n\n{prompt}"

                for chunk in client.models.generate_content_stream(
                    model="gemini-2.5-flash-preview-04-17",
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=4000,
                        temperature=0.2,
                    )
                ):
                    if chunk.text:
                        full_response += chunk.text
                        if show_raw:
                            stream_box.markdown(
                                f"""<div class='stream-container'>{full_response[-2000:]}</div>""",
                                unsafe_allow_html=True
                            )
                
                status.update(label="Analysis complete!", state="complete")
                
            except Exception as e:
                st.error(f"API error: {e}")
                status.update(label="Analysis failed", state="error")
                st.stop()
        
        # Store in session state
        key = f"results_{selected_user['user_id']}"
        st.session_state[key] = {"raw": full_response, "user": selected_user}
        
        patterns = parse_patterns_from_response(full_response)
        st.session_state[f"patterns_{selected_user['user_id']}"] = patterns

# Display stored results
result_key = f"results_{selected_user['user_id']}"
pattern_key = f"patterns_{selected_user['user_id']}"

if result_key in st.session_state:
    full_response = st.session_state[result_key]["raw"]
    patterns = st.session_state.get(pattern_key, [])
    
    # Extract reasoning trace
    trace_match = re.search(r'<reasoning_trace>(.*?)</reasoning_trace>', full_response, re.DOTALL)
    
    # Stats
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-number'>{len(patterns)}</div>
            <div class='stat-label'>Patterns Found</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        high_conf = sum(1 for p in patterns if p.get("confidence") == "high")
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-number' style='color: #22c55e;'>{high_conf}</div>
            <div class='stat-label'>High Confidence</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        strong_ev = sum(1 for p in patterns if p.get("evidence_strength") == "strong")
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-number' style='color: #f59e0b;'>{strong_ev}</div>
            <div class='stat-label'>Strong Evidence</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        sessions_count = len(selected_user["conversations"])
        st.markdown(f"""<div class='stat-card'>
            <div class='stat-number' style='color: #7c3aed;'>{sessions_count}</div>
            <div class='stat-label'>Sessions Analyzed</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Reasoning trace expander
    if trace_match:
        with st.expander("LLM Reasoning Trace (what the system considered)", expanded=False):
            st.markdown(f"""<div class='trace-block'>{trace_match.group(1).strip()}</div>""", unsafe_allow_html=True)
    
    # Pattern cards
    if patterns:
        st.markdown(f"### Detected Patterns")
        
        for p in patterns:
            conf = p.get("confidence", "low")
            conf_color = confidence_color(conf)
            ev_icon = evidence_label(p.get("evidence_strength", "weak"))
            sessions_str = ", ".join(p.get("sessions_involved", []))
            gap = p.get("temporal_gap_days", 0)
            
            st.markdown(f"""
            <div class='pattern-card'>
                <div class='pattern-title'>{ev_icon} {p.get('title', 'Pattern')}</div>
                <div class='pattern-meta'>
                    <span class='badge badge-sessions'>Sessions: {sessions_str}</span>
                    <span class='badge badge-confidence' style='color: {conf_color}; border-color: {conf_color}40; background: {conf_color}15;'>
                        {conf.upper()} CONFIDENCE
                    </span>
                    {f"<span class='badge' style='background: rgba(124,58,237,0.15); color: #c4b5fd; border: 1px solid rgba(124,58,237,0.3);'>{gap}d gap</span>" if gap else ""}
                    <span class='badge' style='background: rgba(100,116,139,0.15); color: #94a3b8; border: 1px solid rgba(100,116,139,0.3);'>{p.get('evidence_strength','?').upper()}</span>
                </div>
                
                <div class='cause-effect'>
                    <div class='cause-box'>{p.get('trigger','?')}</div>
                    <div class='arrow-box'>→</div>
                    {f"<div class='temporal-chip'>{gap}d gap</div><div class='arrow-box'>→</div>" if gap > 1 else ""}
                    <div class='effect-box'>{p.get('symptom','?')}</div>
                </div>
                
                <div style='margin-top: 12px;'>
                    <div style='font-size: 13px; color: #94a3b8; margin-bottom: 6px;'>Temporal Reasoning</div>
                    <div style='font-size: 14px; color: #cbd5e1; line-height: 1.6;'>{p.get('temporal_reasoning','')}</div>
                </div>
                
                <div class='rec-box'>
                    <b>Recommendation:</b> {p.get('recommendation','')}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show confidence justification + reasoning trace in expander
            with st.expander(f"Reasoning trace: {p.get('title','Pattern')}", expanded=False):
                st.markdown(f"""<div class='trace-block'>{p.get('reasoning_trace','No trace available')}</div>""", unsafe_allow_html=True)
                st.markdown(f"""
                <div style='background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2); border-radius: 8px; padding: 12px 16px; margin-top: 8px;'>
                    <span style='color: #22c55e; font-size: 12px; font-family: JetBrains Mono;'>CONFIDENCE JUSTIFICATION: </span>
                    <span style='color: #86efac; font-size: 13px;'>{p.get('confidence_justification','')}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # JSON output
    if show_json and patterns:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### JSON Output (Confidence Scores)")
        
        json_output = []
        for p in patterns:
            json_output.append({
                "pattern_id": p.get("pattern_id"),
                "title": p.get("title"),
                "sessions_involved": p.get("sessions_involved"),
                "trigger": p.get("trigger"),
                "symptom": p.get("symptom"),
                "temporal_gap_days": p.get("temporal_gap_days"),
                "confidence": p.get("confidence"),
                "confidence_justification": p.get("confidence_justification"),
                "evidence_strength": p.get("evidence_strength"),
                "recommendation": p.get("recommendation")
            })
        
        st.markdown(f"""<div class='json-output'>{json.dumps(json_output, indent=2)}</div>""", unsafe_allow_html=True)
        
        # Download button
        st.download_button(
            "Download JSON",
            data=json.dumps(json_output, indent=2),
            file_name=f"clary_patterns_{selected_user['user_id']}.json",
            mime="application/json"
        )

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #334155; font-size: 12px; font-family: JetBrains Mono; border-top: 1px solid #1e2a45; padding-top: 16px;'>
    Clary · Ask First Health Reasoning Engine · Powered by Gemini 2.5 Flash
</div>
""", unsafe_allow_html=True)
