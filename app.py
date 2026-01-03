import streamlit as st
from langchain_core.messages import HumanMessage

# import your existing graph
from chatbot_graph import graph   # adjust filename if needed

st.set_page_config(page_title="Internal AI Chatbot", layout="centered")
st.title("🤖 Internal AI Chatbot")

# session state for conversation
if "state" not in st.session_state:
    st.session_state.state = None

# render previous messages
if st.session_state.state:
    for msg in st.session_state.state["messages"]:
        role = "user" if msg.type == "human" else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

# input box
user_input = st.chat_input("Ask a question...")

if user_input:
    # show user message immediately
    with st.chat_message("user"):
        st.write(user_input)

    # update LangGraph state
    if st.session_state.state is None:
        st.session_state.state = {
            "messages": [HumanMessage(content=user_input)]
        }
    else:
        st.session_state.state["messages"].append(
            HumanMessage(content=user_input)
        )

    # run chatbot
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            st.session_state.state = graph.invoke(
                st.session_state.state
            )
            response = st.session_state.state["messages"][-1].content
            st.write(response)