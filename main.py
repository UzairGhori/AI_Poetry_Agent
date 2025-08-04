import streamlit as st
import asyncio
from agents import Agent, Runner, function_tool
from dotenv import load_dotenv
import requests
from connection import config
import os

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

# --- TOOL FUNCTION ---
@function_tool
def get_poemist_poem() -> str:
    """Fetch a poem using Poemist API (requires key)."""
    headers = {"Authorization": f"Bearer {gemini_api_key}"}
    res = requests.get("https://www.poemist.com/api/v1/randompoems", headers=headers)
    poem = res.json()[0]
    return f"{poem['title']} by {poem['poet']['name']}\n\n{poem['content']}"

# --- AGENTS ---
Lyric_poetry = Agent(
    name="lyric-poetry-agent",
    instructions="You are a lyric poetry expert. Analyze deep emotions and personal reflections in stanzas."
)

Narrative_poetry = Agent(
    name="narrative-poetry-agent",
    instructions="You are a narrative poetry expert. Explain storytelling elements in the poem."
)

Dramatic_poetry = Agent(
    name="dramatic-poetry-agent",
    instructions="You are a dramatic poetry expert. Analyze theatrical and emotional expressions in stanzas."
)

Poetry_Agent = Agent(
    name="poetry-agent",
    instructions="""
        You are a creative poetry assistant. Analyze or generate poems.
        - For lyric poetry, delegate to the lyric poetry agent.
        - For narrative poetry, delegate to the narrative poetry agent.
        - For dramatic poetry, delegate to the dramatic poetry agent.
        - For analysis, use the analyzer_poem tool.
        - You can also always reply in Roman Urdu if needed.
    """,
    tools=[get_poemist_poem],
    handoffs=[Lyric_poetry, Narrative_poetry, Dramatic_poetry],
)

# --- ASYNC HANDLER ---
async def analyze_poetry(prompt):
    result = await Runner.run(
        starting_agent=Poetry_Agent,
        input=prompt,
        run_config=config
    )
    return result.final_output

# --- CUSTOM STYLING ---
st.markdown("""
    <style>
        body {
            font-family: 'Segoe UI', sans-serif;
        }
        .title-section {
            background: linear-gradient(90deg, #7b2ff7, #f107a3);
            border-radius: 12px;
            padding: 2rem;
            color: white;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        .stTextArea textarea {
            background-color: black !important;
            border-radius: 10px !important;
            border: 1px solid #d1d5db !important;
        }
        .stSelectbox, .stButton button {
            font-size: 1rem;
        }
        .footer {
            text-align: center;
            color: #6b7280;
            margin-top: 3rem;
            font-size: 0.9rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.set_page_config(page_title="AI Poetry Analyzer", page_icon="🧠", layout="wide")
st.markdown("""
<div class="title-section">
    <h1>🧠 AI Poetry Analyzer</h1>
    <p>Explore emotions, structure & meaning in your poetry with expert AI agents</p>
</div>
""", unsafe_allow_html=True)

# --- UI LAYOUT ---
st.subheader("🔍 Poetry Mode")
poetry_type = st.selectbox("Choose the Poetry Category", ["General", "Lyric", "Narrative", "Dramatic"])

with st.expander("✍️ Write or Paste Your Poem / Query"):
    user_input = st.text_area("Your poetic text", placeholder="Paste or write your poem here...", height=220)

# --- ANALYZE ---
if st.button("🎯 Analyze / Generate", use_container_width=True):
    if not user_input.strip():
        st.warning("⚠️ Please enter a query or poem to continue.")
    else:
        with st.spinner("🔎 Processing with expert agents..."):
            full_prompt = user_input
            if poetry_type != "General":
                full_prompt = f"Please analyze this {poetry_type.lower()} poetry:\n\n{user_input}"
            result = asyncio.run(analyze_poetry(full_prompt))
            st.success("✅ Analysis Complete!")

            st.markdown("### 🧾 AI Response")
            st.markdown(f"```text\n{result}\n```")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    Build in with ❤️ by Uzair Ghori | Powered by OpenAI SDK + Streamlit + Python
</div>
""", unsafe_allow_html=True)
