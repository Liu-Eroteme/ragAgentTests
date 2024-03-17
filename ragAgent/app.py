import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

st.set_page_config(page_title="ChatPDF")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if (
        st.session_state["user_input"]
        and len(st.session_state["user_input"].strip()) > 0
    ):
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def load_knowledge_base():
    with st.session_state["ingestion_spinner"], st.spinner("Loading knowledge base"):
        st.session_state["assistant"].ingest()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""


def page():
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatPDF()
    if "ingestion_spinner" not in st.session_state:
        st.session_state["ingestion_spinner"] = st.empty()

    st.header("ChatPDF")

    if st.button("Load Knowledge Base"):
        load_knowledge_base()

    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)


if __name__ == "__main__":
    page()
