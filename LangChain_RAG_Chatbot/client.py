import requests
import streamlit as st


def get_groq_response(input_text):
    json_body={"input": input_text}
    headers = {'Content-Type': 'application/json'}  # Add Content-Type header
    response=requests.post("http://127.0.0.1:8000/invoke", json=json_body, headers=headers)
    # print(response.json())

    return response.json()

## Streamlit app
st.title("Attention mechanism QA Chatbot")
st.write("This chatbot uses the LangChain RAG model to answer questions related to attention mechanism in transformer architecture")
input_text=st.text_input("Please enter your question below and the chatbot will provide an answer")


if input_text:
    with st.spinner("Generating answer..."):
        answer_data = get_groq_response(input_text)
        # print(answer_data)

    st.subheader("Answer:")
    st.write(answer_data["answer"])

    st.subheader("Sources:")
    st.write("Chatbot used the below page content as context from retriever to answer your question:")
    if "sources" in answer_data and answer_data["sources"]:
        for source in answer_data["sources"]:
            st.write(source)

    else:
        st.write("No sources found")
