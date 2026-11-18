import requests
import streamlit as st

from chatbot_ui.core.config import config


def api_call(method, url, **kwargs):

    def _show_error_popup(message):
        """Show error message as a popup in the top-right corner."""
        st.session_state["error_popup"] = {
            "visible": True,
            "message": message,
        }

    try:
        response = getattr(requests, method)(url, **kwargs)

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            response_data = {"message": "Invalid response format from server"}

        if response.ok:
            return True, response_data

        return False, response_data

    except requests.exceptions.ConnectionError:
        _show_error_popup("Connection error. Please check your network connection.")
        return False, {"message": "Connection error"}
    except requests.exceptions.Timeout:
        _show_error_popup("The request timed out. Please try again later.")
        return False, {"message": "Request timeout"}
    except Exception as e:
        _show_error_popup(f"An unexpected error occurred: {str(e)}")
        return False, {"message": str(e)}

with st.sidebar:
    provider = st.selectbox("Provider", ["OpenAI", "Anthropic", "Google"])
    if provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-6-luna", "gpt-6-sol"])
    elif provider == "Anthropic":
        model_name = st.selectbox("Model", ["claude-haiku-5-5", "claude-sonnet-5-5"])
    elif provider == "Google":
        model_name = st.selectbox("Model", ["gemini-3.6-flash", "gemini-3.8-flash", "gemini-4-hydrogen"])
    else:
        raise ValueError(f"Invalid provider: {provider}")
    
    st.session_state.provider = provider
    st.session_state.model_name = model_name


if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today?"}]


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # skip the canned greeting so every request starts with a user turn
        history = st.session_state.messages[1:]
        ok, response_data = api_call("post", f"{config.API_URL}/chat", json={"provider": st.session_state.provider, "models_name": st.session_state.model_name, "messages": history})
        if ok:
            answer = response_data["message"]
            st.write(answer)
        else:
            # FastAPI errors use "detail"; api_call's own errors use "message"
            st.write(response_data.get("detail") or response_data.get("message") or "Request failed")
    if ok:
        st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        # drop the unanswered user turn so a retry doesn't send two user messages in a row
        st.session_state.messages.pop()
    