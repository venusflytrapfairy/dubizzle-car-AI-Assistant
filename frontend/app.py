"""
Streamlit client for the dubizzle Car Assistant.

Talks to the FastAPI backend over HTTP only (clean client/server boundary -
this file contains zero business logic, LLM calls, or DB access; it just
renders chat state and forwards messages to /chat).
"""
import os

import httpx
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

DUBIZZLE_RED = "#E00201"

st.set_page_config(page_title="dubizzle Cars Assistant", page_icon="🚗", layout="centered")

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: #FFFFFF; }}
    section[data-testid="stSidebar"] {{ background-color: #FAFAFA; border-right: 1px solid #eee; }}
    .dz-header {{
        background-color: {DUBIZZLE_RED};
        color: white;
        padding: 18px 24px;
        border-radius: 10px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 12px;
    }}
    .dz-header h1 {{ font-size: 1.4rem; margin: 0; color: white; }}
    .dz-header p {{ margin: 0; font-size: 0.85rem; opacity: 0.9; }}
    .stChatMessage {{ border-radius: 10px; }}
    div[data-testid="stChatMessage"] a {{ color: {DUBIZZLE_RED}; }}
    .stButton>button {{
        background-color: {DUBIZZLE_RED};
        color: white;
        border: none;
        border-radius: 8px;
    }}
    .stButton>button:hover {{ background-color: #b40201; color: white; }}
    .car-card {{
        border: 1px solid #eee;
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 8px;
        background-color: #fff8f8;
        border-left: 4px solid {DUBIZZLE_RED};
    }}
    .car-card b {{ color: {DUBIZZLE_RED}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="dz-header">
        <div style="font-size:2rem;">🚗</div>
        <div>
            <h1>dubizzle Cars Assistant</h1>
            <p>Ask me about listings, book a viewing, or just say hi</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- state ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "greeted" not in st.session_state:
    st.session_state.greeted = False

# ------------------------------------------------------------- sidebar -----
with st.sidebar:
    st.subheader("👤 Your profile")
    st.caption("Enter any name/ID - use the same one next time to test long-term memory recall.")
    user_id_input = st.text_input("Your name or user ID", value=st.session_state.user_id or "")

    col1, col2 = st.columns(2)
    with col1:
        start_clicked = st.button("Start / Switch user", use_container_width=True)
    with col2:
        new_session_clicked = st.button("New session (same user)", use_container_width=True)

    if start_clicked and user_id_input.strip():
        st.session_state.user_id = user_id_input.strip()
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.greeted = False
        st.rerun()

    if new_session_clicked and st.session_state.user_id:
        st.session_state.session_id = None
        st.session_state.messages = []
        st.session_state.greeted = False
        st.rerun()

    st.divider()
    st.caption(
        "💡 Tip: chat as 'sara', book a viewing, then click **New session** "
        "(or reload and re-enter 'sara') to see the agent recall your preferences "
        "as a brand-new conversation."
    )

    if st.session_state.user_id:
        st.divider()
        st.subheader("🧠 Long-term memory (debug)")
        try:
            resp = httpx.get(f"{BACKEND_URL}/users/{st.session_state.user_id}/profile", timeout=5)
            if resp.status_code == 200:
                st.json(resp.json())
            else:
                st.caption("No profile yet - say hello first!")
        except httpx.HTTPError:
            st.caption("⚠️ Backend not reachable.")

# ------------------------------------------------------------ main pane ----
if not st.session_state.user_id:
    st.info("👈 Enter your name in the sidebar and click **Start / Switch user** to begin.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for tc in msg.get("tool_calls", []):
            if tc["tool"] == "search_inventory" and tc["result"].get("listings"):
                for car in tc["result"]["listings"]:
                    st.markdown(
                        f"""<div class="car-card">
                        <b>#{car['listing_id']} {car['year']} {car['make'].title()} {car['model'].title()}</b><br/>
                        {car['price_display']} · {car.get('body_type','')} · {car.get('mileage_km') and f"{car['mileage_km']:,} km" or "mileage n/a"}
                        </div>""",
                        unsafe_allow_html=True,
                    )

prompt = st.chat_input("Ask about a car, book a viewing, or say hi...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking the inventory..."):
            try:
                resp = httpx.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "user_id": st.session_state.user_id,
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                st.session_state.session_id = data["session_id"]
                reply = data["reply"]
                tool_calls = data.get("tool_calls", [])
            except httpx.HTTPError as e:
                reply = f"⚠️ Couldn't reach the backend ({e}). Is the FastAPI server running?"
                tool_calls = []

        st.markdown(reply)
        for tc in tool_calls:
            if tc["tool"] == "search_inventory" and tc["result"].get("listings"):
                for car in tc["result"]["listings"]:
                    st.markdown(
                        f"""<div class="car-card">
                        <b>#{car['listing_id']} {car['year']} {car['make'].title()} {car['model'].title()}</b><br/>
                        {car['price_display']} · {car.get('body_type','')} · {car.get('mileage_km') and f"{car['mileage_km']:,} km" or "mileage n/a"}
                        </div>""",
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({"role": "assistant", "content": reply, "tool_calls": tool_calls})

if not st.session_state.messages:
    st.caption("Try: \"Show me SUVs under 150k\", \"Any Mercedes with warranty?\", \"Book a viewing for #3 on Monday at 10am\"")
