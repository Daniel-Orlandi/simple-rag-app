import os
import requests
import uuid
import logging
from dotenv import load_dotenv

import streamlit as st
from src.utils.logging_config import setup_logging

from src.models.config import RAGConfig

load_dotenv()

API_URL = os.getenv("API_URL")

logger = logging.getLogger(__name__)

# Initialize RAG configuration
rag_config = RAGConfig()
available_providers = rag_config.available_providers
available_language_models = rag_config.available_language_models


st.set_page_config(page_title="Document AI Q&A Assistant", page_icon="📚", layout="wide")

if "user_id_key" not in st.session_state:
    st.session_state.user_id_key = str(uuid.uuid4())

# Setup logging
if "logging_initialized" not in st.session_state:
    log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    log_user_id = os.getenv("LOG_USER_ID", st.session_state.user_id_key)
    st.session_state.logging_initialized = setup_logging(
        level=log_level,
        user_id=log_user_id,
        session_id=str(uuid.uuid4())
    )
    logger.info(f"Logging Setup Complete for user: {log_user_id}")

# Initialize session state
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""

if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = ""

logger.info("Starting Document AI Q&A Assistant")

st.title("📚 Document AI Q&A Assistant")

# Sidebar
with st.sidebar:
    # ============== Model Settings ==============
    st.header("Model Settings")
    
    # Provider selection
    provider = st.selectbox(
        "Provider",
        options=list(available_providers.keys()),
        format_func=lambda x: f"{available_providers[x]['name']}",
        index=None     
    )

    if not provider:
        st.warning("Please select a model provider")
        st.stop()

    
    # ============== API Keys ==============
    st.header("API Keys")

    if provider == "groq":
        # Groq API Key
        with st.expander("Groq API Key", expanded=(provider == "groq" and not st.session_state.groq_api_key)):
            st.caption(f"[Get your free key]({available_providers['groq']['get_key_url']})")
            groq_key = st.text_input(
                "Groq API Key",
                type="password",
                value=st.session_state.groq_api_key,
                key="groq_key_input",
                label_visibility="collapsed",
                placeholder="gsk_..."
            )
            if groq_key != st.session_state.groq_api_key:
                st.session_state.groq_api_key = groq_key

            st.session_state.current_api_key = st.session_state.groq_api_key
            

    elif provider == "gemini":         
    
        # Gemini API Key
        with st.expander("Gemini API Key", expanded=(provider == "gemini" and not st.session_state.gemini_api_key)):
            st.caption(f"[Get your free key]({available_providers['gemini']['get_key_url']})")
            gemini_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.gemini_api_key,
                key="gemini_key_input",
                label_visibility="collapsed",
                placeholder="AIza..."
            )
            if gemini_key != st.session_state.gemini_api_key:
                st.session_state.gemini_api_key = gemini_key
            
            st.session_state.current_api_key = st.session_state.gemini_api_key    

        
    # Model selection based on provider
    model_options = available_language_models[provider]
    model = st.selectbox(
        "Model",
        options=[m[0] for m in model_options],
        format_func=lambda x: next(m[1] for m in model_options if m[0] == x)
    )
    
    # Temperature slider
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
    
    st.divider()
    
    
    
    st.divider()
    
    # Document Upload 
    st.header("📄 Upload Documents")
    MAX_FILES = 10

    uploaded_files = st.file_uploader(
        f"Upload up to {MAX_FILES} PDF or HTML files",
        type=["pdf", "html"],
        accept_multiple_files=True,
        key=st.session_state.user_id_key,
        disabled=st.session_state.is_processing
    )

    if uploaded_files:
        # Filter to only new files
        new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files]
        
        if new_files:
            if len(new_files) > MAX_FILES:
                st.error(f"Only {MAX_FILES} files can be uploaded at a time.")
                new_files = new_files[:MAX_FILES]

            if st.button("Process Documents", type="primary"):
                st.session_state.is_processing = True
                
                with st.spinner("Uploading and processing documents..."):
                    # Prepare files for multipart upload
                    files_to_upload = [
                        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                        for f in new_files
                    ]
                    
                    try:
                        response = requests.post(f"{API_URL}/upload",
                                                 params={"session_id": st.session_state.user_id_key},
                                                 files=files_to_upload)
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ {result['message']}")
                            logger.debug(f"Documents indexed: {result['documents_indexed']}")
                            logger.debug(f"Total chunks: {result['total_chunks']}")
                            
                            # Track uploaded files
                            st.session_state.uploaded_files.extend([f.name for f in new_files])
                            logger.debug(f"Uploaded files: {st.session_state.uploaded_files}")
                           
                        else:
                            st.error(f"Failed to process: {response.text}")
                            logger.error(f"Upload failed: {response.status_code} {response.text}")

                    except requests.exceptions.ConnectionError as connection_error:
                        st.error(f"Cannot connect to API. Make sure the server is running on {API_URL}.")
                        logger.error(f"Cannot connect to API: {connection_error}")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Upload error: {e}", exc_info=True)
                    finally:
                        st.session_state.is_processing = False

    # uploaded files
    if st.session_state.uploaded_files:
        st.divider()
        st.subheader("Loaded Documents")
        for doc in st.session_state.uploaded_files:
            st.text(f"• {doc}")
       

# Main chat area
st.header(" Ask Questions")
st.markdown("Ask questions about the documents you uploaded.")

# chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("model"):
            model_label = next((m[1] for m in available_language_models.get(message.get("provider", ""), []) if m[0] == message["model"]), message["model"])
            st.caption(f"Model: {model_label}")
        if "references" in message:
            with st.expander("📖 View References"):
                for i, ref in enumerate(message["references"], 1):
                    st.markdown(f"**Reference {i}:**")
                    st.text(ref[:500] + "..." if len(ref) > 500 else ref)
                    st.divider()


if prompt := st.chat_input("Ask a question about your documents..."):    
    
    # Add user message    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    logger.debug(f"User message: {prompt}")

    # Generate response
    with st.chat_message("assistant"):
        # Check if API key is configured
        if not st.session_state.current_api_key:
            error_msg = f"Please enter your {available_providers[provider]['name']} API key in the sidebar."
            st.warning(error_msg)
            st.caption(f"Model: {next(m[1] for m in available_language_models[provider] if m[0] == model)}")
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "model": model, "provider": provider})
            logger.warning(error_msg)

        else:
            with st.spinner(f"Thinking ({model})..."):
                try:
                    response = requests.post(
                        f"{API_URL}/question",
                        json={
                            "session_id":st.session_state.user_id_key,
                            "question": prompt,
                            "provider": provider,
                            "model": model,
                            "api_key": st.session_state.current_api_key,
                            "temperature": temperature
                        },
                        headers={"Content-Type": "application/json"}
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        answer = result["answer"]
                        references = result["references"]
                        
                        st.markdown(answer)
                        st.caption(f"Model: {next(m[1] for m in available_language_models[provider] if m[0] == model)}")
                        
                        if references:
                            with st.expander("References Used"):
                                for i, ref in enumerate(references, 1):
                                    st.markdown(f"**Reference {i}:**")
                                    st.text(ref[:500] + "..." if len(ref) > 500 else ref)
                                    st.divider()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "references": references,
                            "model": model,
                            "provider": provider
                        })

                    elif response.status_code == 401:
                        error_msg = "Invalid API key. Please check your key in the sidebar and try again."
                        st.warning(error_msg)
                        st.caption(f"Model: {next(m[1] for m in available_language_models[provider] if m[0] == model)}")
                        st.session_state.messages.append({"role": "assistant", "content": error_msg, "model": model, "provider": provider})
                        logger.warning(f"Auth error: {response.text}")

                    else:
                        error_msg = f"Error: {response.text}"
                        st.error(error_msg)
                        st.caption(f"Model: {next(m[1] for m in available_language_models[provider] if m[0] == model)}")
                        st.session_state.messages.append({"role": "assistant", "content": error_msg, "model": model, "provider": provider})
                        logger.error(f"Query error: {error_msg}")

                except requests.exceptions.ConnectionError as e:
                    error_msg = f"Cannot connect to API. Make sure the server is running on {API_URL}."
                    st.error(error_msg)
                    st.caption(f"Model: {next(m[1] for m in available_language_models[provider] if m[0] == model)}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "model": model, "provider": provider})
                    logger.error(f"Connection error: {e}")

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.caption(f"Model: {next(m[1] for m in available_language_models[provider] if m[0] == model)}")
                    st.session_state.messages.append({"role": "assistant", "content": error_msg, "model": model, "provider": provider})
                    logger.error(f"Query error: {e}")