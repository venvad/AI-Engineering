import logging
from typing import Any

import requests
import streamlit as st

from chatbot_ui.core.config import config

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 60


def api_call(method: str, url: str, **kwargs: Any) -> tuple[bool, dict[str, Any]]:
    kwargs.setdefault("timeout", DEFAULT_TIMEOUT)

    try:
        response = requests.request(method, url, **kwargs)
    except requests.exceptions.Timeout:
        logger.warning("Request to %s timed out", url)
        return False, {"message": "The request timed out. Please try again later."}
    except requests.exceptions.ConnectionError:
        logger.exception("Connection error calling %s", url)
        return False, {
            "message": "Connection error. Please check your network connection."
        }
    except requests.exceptions.RequestException:
        logger.exception("Request failed calling %s", url)
        return False, {"message": "Failed to reach the server. Please try again."}
    else:
        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            logger.warning("Non-JSON response (%s) from %s", response.status_code, url)
            return False, {"message": "Invalid response format from server"}

        if not response.ok:
            logger.warning(
                "API error (%s) from %s: %s", response.status_code, url, response_data
            )
            return False, response_data

        return True, response_data


with st.sidebar:
    provider = st.selectbox("Provider", ["OpenAI", "Anthropic", "Google"])
    if provider == "OpenAI":
        model_name = st.selectbox("Model", ["gpt-6-sol", "gpt-6-luna"])
    elif provider == "Anthropic":
        model_name = st.selectbox("Model", ["claude-haiku-5-5", "claude-sonnet-5-5"])
    elif provider == "Google":
        model_name = st.selectbox("Model", ["gemini-4-hydrogen", "gemini-4-neon"])
    else:
        raise ValueError(f"Invalid provider: {provider}")

    st.session_state.provider = provider
    st.session_state.model_name = model_name


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.chat_message("assistant"):
    st.markdown("Hello! How can I assist you today?")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Hello! How can I assist you today?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    answer = ""
    with st.chat_message("assistant"):
        ok, response_data = api_call(
            "post",
            f"{config.API_URL}/rag",
            json={
                "query": prompt,
                "provider": st.session_state.provider,
                "model_name": st.session_state.model_name,
                "history": st.session_state.messages,
            },
        )
        if ok:
            answer = response_data["answer"]
            st.write(answer)
        else:
            # FastAPI errors use "detail"; api_call's own errors use "message"
            st.write(
                response_data.get("detail")
                or response_data.get("message")
                or "Request failed"
            )
    if ok:
        st.session_state.messages.extend(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        )
