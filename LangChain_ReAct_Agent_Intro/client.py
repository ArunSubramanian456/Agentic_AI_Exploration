import streamlit as st
import requests
import json

## Streamlit app
st.set_page_config(page_title="LangChain AI Agent Chat", page_icon="💬", layout="wide")
st.title(" LangChain AI Agent Chat")

query = st.chat_input("Please enter your question ")

if query:
    st.session_state.messages= []
    st.session_state.full_response = ""

    st.chat_message("user").write(query)
    assistant_msg = st.chat_message("assistant")
    response_container = assistant_msg.container()

    # Function to handle the streaming response from FastAPI
    def get_streaming_response(query: str):
        payload = {"query": query}
        headers = {"Content-Type": "application/json"}
        with requests.post("http://localhost:8000/stream_chat", json=payload,headers=headers,  stream=True, timeout = 30) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                if chunk:
                    try:
                        # FastAPI sends each chunk as a separate JSON object
                        data = json.loads(chunk)
                        if data["type"] == "agent_action":
                                yield f"""  \n\n**{data['tool']}**: {data['tool_input']}  \n\n {data['log']} """                     
                        elif data["type"] == "final_answer":
                            yield f"""  \n\n**Final Answer**\n\n {data["output"]}"""
                        elif data["type"] == "error":
                            yield f"  \n\n:red[Error]: {data['error']}  \n\n"
                    except json.JSONDecodeError:
                        yield chunk
                
    # Stream the response from FastAPI
    for response_chunk in get_streaming_response(query):
        if isinstance(response_chunk, dict) and "output" in response_chunk:
            st.session_state.full_response += str(response_chunk["output"])
        else:
            st.session_state.full_response += str(response_chunk)
        st.session_state.messages.append({"role": "assistant", "content": response_chunk})
        response_container.write(st.session_state.full_response)

