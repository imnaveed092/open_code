import streamlit as st
from new_utiles import process_query

# Page Configuration
st.set_page_config(page_title="Regulation Information Retrieval", page_icon="📘")
st.title("Regulation Information Retrieval (PoC)")

# Sidebar: Use case and LLM selection
st.sidebar.title("Options")
use_case = st.sidebar.selectbox("Choose Use Case", ["Chat with Bot", "Flint Output"])
llm_model_name = st.sidebar.selectbox(
    "Choose LLM",
    ["llama-3.1-70b-instruct", "LLM2", "LLM3"],
    index=0,
    key="llmModelName"
)

# User query input
query = st.chat_input(placeholder="Enter your query...")

# Submit button
# if st.button("Retrieve Regulation"):
if query:
    with st.spinner("Retrieving information..."):
        try:
            # Process the query using the backend
            response = process_query(use_case, llm_model_name, query)

            # Update session history
            if "history" not in st.session_state:
                st.session_state["history"] = []
            st.session_state["history"].append({"query": query, "response": response})

            # Display the response
            st.chat_message("assistant").write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.error("Please enter a query.")

# Display session history
if "history" in st.session_state and st.session_state["history"]:
    st.sidebar.subheader("Session History")
    for idx, interaction in enumerate(st.session_state["history"], start=1):
        st.sidebar.write(f"**Query {idx}:** {interaction['query']}")
        st.sidebar.write(f"**Response {idx}:** {interaction['response']}")
        st.sidebar.write("---")
