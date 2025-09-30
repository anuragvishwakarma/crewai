# app.py
import streamlit as st
from crew.crew_manager import run_crew

st.set_page_config(page_title="FieldOps Assistant (CrewAI + Ollama)", layout="wide")
st.title("🛠️ FieldOps Multi-Agent Assistant")
st.caption("Powered by CrewAI, Llama3, and Ollama")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "agent" in msg:
            st.caption(f".Handled by: {msg['agent']}")

if prompt := st.chat_input("Ask about maintenance, support, or workload..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consulting experts..."):
            result = run_crew(prompt, st.session_state.messages)
            st.markdown(result["response"])
            st.caption(f".Handled by: {result['agent_used']}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": result["response"],
                "agent": result["agent_used"]
            })