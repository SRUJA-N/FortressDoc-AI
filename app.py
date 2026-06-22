"""
FortressDoc AI
---------------
A single-page Streamlit application for uploading a PDF, extracting its text,
chunking the document with LangChain, and asking questions through a chat UI.

Run with:
    streamlit run app.py

Expected packages:
    streamlit
    langchain
    langchain-text-splitters
    pdfplumber  # preferred PDF extractor
    PyPDF2      # fallback PDF extractor
"""

from __future__ import annotations

import io
import logging
import os
import base64
import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode

import streamlit as st
import requests

# LangChain's text splitter has moved between packages across releases. Prefer
# the current package, but keep the older import path so the app remains usable
# if someone installs an older LangChain distribution.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Application-level configuration
# ---------------------------------------------------------------------------

APP_TITLE = "FortressDoc AI"
CHUNK_SIZE = 1_200
CHUNK_OVERLAP = 200
MAX_CONTEXT_CHUNKS = 6
CHUTES_AUTH_BASE_URL = "https://api.chutes.ai"
CHUTES_DEFAULT_OAUTH_SCOPES = "openid profile chutes:invoke"
OAUTH_STATE_FILE = Path(__file__).with_name(".chutes_oauth_states.json")
OAUTH_STATE_TTL_SECONDS = 10 * 60

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Streamlit page configuration and styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        /* Keep the UI restrained and corporate without hiding Streamlit affordances. */
        .block-container {
            max-width: 1120px;
            padding-top: 2.25rem;
            padding-bottom: 3rem;
        }

        [data-testid="stSidebar"] {
            background: #0f172a;
        }

        [data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.24);
            background: #1e293b;
            color: #f8fafc;
            font-weight: 600;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            border-color: #93c5fd;
            color: #ffffff;
        }

        .fortress-subtitle {
            color: #475569;
            font-size: 1rem;
            margin-top: -0.75rem;
            margin-bottom: 1.5rem;
        }

        .status-panel {
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            background: #f8fafc;
            margin-bottom: 1.25rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session-state initialization
# ---------------------------------------------------------------------------

def initialize_session_state() -> None:
    """Create all session-state keys used by the app exactly once."""

    defaults = {
        "messages": [],
        "pdf_name": None,
        "pdf_text": "",
        "pdf_chunks": [],
        "processing_error": None,
        "chutes_access_token": None,
        "chutes_refresh_token": None,
        "chutes_token_response": None,
        "chutes_user_profile": None,
        "chutes_oauth_state": None,
        "chutes_pkce_verifier": None,
        "auth_error": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session_state()


# ---------------------------------------------------------------------------
# PDF extraction helpers
# ---------------------------------------------------------------------------

def extract_text_with_pdfplumber(pdf_bytes: bytes) -> str:
    """
    Extract PDF text with pdfplumber.

    pdfplumber generally performs well for text-based PDFs and gives us a
    straightforward page-by-page extraction flow. It is imported lazily so the
    app can still run if the environment only has PyPDF2 installed.
    """

    import pdfplumber

    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(f"\n\n--- Page {page_number} ---\n{page_text}")

    return "\n".join(text_parts).strip()


def extract_text_with_pypdf2(pdf_bytes: bytes) -> str:
    """
    Extract PDF text with PyPDF2 as a fallback.

    This keeps the app usable in lightweight environments where pdfplumber is
    unavailable or fails on a particular PDF structure.
    """

    from PyPDF2 import PdfReader

    reader = PdfReader(io.BytesIO(pdf_bytes))
    text_parts: list[str] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            text_parts.append(f"\n\n--- Page {page_number} ---\n{page_text}")

    return "\n".join(text_parts).strip()


def extract_pdf_text(uploaded_file) -> str:
    """
    Extract text from an uploaded PDF.

    The uploaded Streamlit file object is read once into bytes so either parser
    can consume it safely. pdfplumber is attempted first, then PyPDF2.
    """

    pdf_bytes = uploaded_file.getvalue()

    try:
        return extract_text_with_pdfplumber(pdf_bytes)
    except Exception as pdfplumber_error:
        logger.warning("pdfplumber extraction failed; falling back to PyPDF2.")
        logger.debug("pdfplumber error details", exc_info=pdfplumber_error)

    try:
        return extract_text_with_pypdf2(pdf_bytes)
    except Exception as pypdf2_error:
        logger.exception("All PDF extraction methods failed.")
        raise RuntimeError(
            "FortressDoc AI could not extract text from this PDF. "
            "The file may be scanned, encrypted, corrupted, or image-only."
        ) from pypdf2_error


# ---------------------------------------------------------------------------
# Chunking and context helpers
# ---------------------------------------------------------------------------

def chunk_document_text(text: str) -> list[str]:
    """Split extracted document text into manageable overlapping chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def select_context_for_prompt(chunks: Iterable[str], user_prompt: str) -> str:
    """
    Select a small context window from session-state chunks.

    This intentionally avoids vector databases or external stores. For now, it
    uses a simple keyword overlap score to choose a few likely relevant chunks.
    The dummy Chutes function below returns a placeholder regardless, but this
    prepares the app's control flow for a real API call later.
    """

    prompt_terms = {
        term.strip(".,:;!?()[]{}\"'").lower()
        for term in user_prompt.split()
        if len(term.strip(".,:;!?()[]{}\"'")) > 2
    }

    scored_chunks: list[tuple[int, int, str]] = []
    for index, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        score = sum(1 for term in prompt_terms if term in chunk_lower)
        scored_chunks.append((score, index, chunk))

    # Prefer keyword matches, then preserve document order as a stable tie-break.
    scored_chunks.sort(key=lambda item: (-item[0], item[1]))
    selected_chunks = [chunk for _, _, chunk in scored_chunks[:MAX_CONTEXT_CHUNKS]]

    return "\n\n".join(selected_chunks)


# ---------------------------------------------------------------------------
# Chutes API placeholder
# ---------------------------------------------------------------------------

def query_chutes_api(context: str, user_prompt: str) -> str:
    """
    Send a document-grounded chat request to a Chutes-hosted model.

    Required environment variable:
        CHUTES_API_KEY

    Optional environment variables:
        CHUTES_BASE_URL
            Defaults to the shared Chutes OpenAI-compatible inference endpoint:
            https://llm.chutes.ai/v1

            For a guaranteed TEE node, point this at your own Chutes deployment
            that was deployed with the documented Chute(..., tee=True, ...) flag.
            The Chutes docs define TEE as a chute/deployment option, not as a
            standard per-request chat-completions parameter.

        CHUTES_MODEL
            Defaults to deepseek-ai/DeepSeek-V3-0324, which the Chutes docs use
            in their OpenAI-compatible shared endpoint example.
    """

    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

    api_key = st.session_state.get("chutes_access_token") or os.getenv("CHUTES_API_KEY")
    if not api_key:
        return (
            "Chutes authentication is not configured. Sign in with Chutes or "
            "set CHUTES_API_KEY and try again."
        )

    base_url = os.getenv("CHUTES_BASE_URL", "https://llm.chutes.ai/v1")
    model = os.getenv("CHUTES_MODEL", "deepseek-ai/DeepSeek-V3-0324")

    system_prompt = (
        "You are FortressDoc AI, a careful document-analysis assistant. "
        "Answer the user's question using only the provided document context. "
        "If the answer is not present in the context, say that the document "
        "does not contain enough information to answer confidently."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "Document context:\n"
                f"{context}\n\n"
                "User question:\n"
                f"{user_prompt}"
            ),
        },
    ]

    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=45.0,
        )

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=700,
        )

        answer = response.choices[0].message.content
        return answer.strip() if answer else "Chutes returned an empty response."

    except APITimeoutError:
        return "The Chutes API request timed out. Please try again in a moment."
    except APIConnectionError:
        return (
            "FortressDoc AI could not connect to Chutes. Check your network "
            "connection and CHUTES_BASE_URL."
        )
    except APIStatusError as error:
        return (
            "Chutes returned an API error "
            f"({error.status_code}): {error.response.text}"
        )
    except Exception as error:
        logger.exception("Unexpected Chutes API error.")
        return f"Unexpected Chutes API error: {error}"


# ---------------------------------------------------------------------------
# Chutes OAuth helpers
# ---------------------------------------------------------------------------

def get_oauth_config() -> dict[str, str]:
    """
    Load Chutes OAuth configuration from environment variables.

    Your redirect URI must exactly match one of the redirect URIs registered in
    the Chutes OAuth app. For local Streamlit development, register:
        http://localhost:8501
    """

    return {
        "client_id": os.getenv("CHUTES_OAUTH_CLIENT_ID", ""),
        "client_secret": os.getenv("CHUTES_OAUTH_CLIENT_SECRET", ""),
        "redirect_uri": os.getenv("CHUTES_OAUTH_REDIRECT_URI", "http://localhost:8501"),
        "scope": os.getenv("CHUTES_OAUTH_SCOPES", CHUTES_DEFAULT_OAUTH_SCOPES),
    }


def base64url_encode(raw_bytes: bytes) -> str:
    """Return unpadded base64url text, as required by OAuth PKCE."""

    return base64.urlsafe_b64encode(raw_bytes).decode("ascii").rstrip("=")


def create_pkce_pair() -> tuple[str, str]:
    """
    Create a PKCE code verifier and S256 challenge.

    Chutes documents OAuth 2.0 Authorization Code with PKCE: generate a random
    verifier, send BASE64URL(SHA256(verifier)) as the challenge, then send the
    original verifier during token exchange.
    """

    verifier = base64url_encode(secrets.token_bytes(64))
    challenge = base64url_encode(hashlib.sha256(verifier.encode("ascii")).digest())
    return verifier, challenge


def load_oauth_state_store() -> dict[str, dict]:
    """Load short-lived OAuth state records from local server-side storage."""

    if not OAUTH_STATE_FILE.exists():
        return {}

    try:
        with OAUTH_STATE_FILE.open("r", encoding="utf-8") as state_file:
            data = json.load(state_file)
            return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        logger.exception("Unable to read Chutes OAuth state store.")
        return {}


def save_oauth_state_store(state_store: dict[str, dict]) -> None:
    """Persist OAuth state records to a local file next to the Streamlit app."""

    try:
        with OAUTH_STATE_FILE.open("w", encoding="utf-8") as state_file:
            json.dump(state_store, state_file)
    except OSError:
        logger.exception("Unable to write Chutes OAuth state store.")


def prune_oauth_state_store(state_store: dict[str, dict]) -> dict[str, dict]:
    """Remove expired OAuth state records so stale sign-in attempts cannot linger."""

    now = time.time()
    return {
        state: payload
        for state, payload in state_store.items()
        if now - float(payload.get("created_at", 0)) <= OAUTH_STATE_TTL_SECONDS
    }


def store_oauth_state(state: str, code_verifier: str) -> None:
    """
    Store the PKCE verifier server-side, keyed by OAuth state.

    Streamlit session state is convenient but can be lost across a full external
    OAuth redirect. This short-lived local store lets the callback safely recover
    the verifier without exposing it in the browser URL.
    """

    state_store = prune_oauth_state_store(load_oauth_state_store())
    state_store[state] = {
        "code_verifier": code_verifier,
        "created_at": time.time(),
    }
    save_oauth_state_store(state_store)


def pop_oauth_code_verifier(state: str) -> str | None:
    """Return and delete the PKCE verifier for a callback state value."""

    state_store = prune_oauth_state_store(load_oauth_state_store())
    payload = state_store.pop(state, None)
    save_oauth_state_store(state_store)

    if not payload:
        return None

    code_verifier = payload.get("code_verifier")
    return code_verifier if isinstance(code_verifier, str) else None


def create_chutes_authorization_url() -> str | None:
    """
    Build the Chutes authorization URL and store CSRF/PKCE values in session.

    Returns None when required client configuration is missing, allowing the UI
    to show a clear setup error instead of a broken login link.
    """

    config = get_oauth_config()
    if not config["client_id"]:
        st.session_state.auth_error = (
            "CHUTES_OAUTH_CLIENT_ID is not set. Register a Chutes OAuth app "
            "and add its client ID to your environment."
        )
        return None

    verifier, challenge = create_pkce_pair()
    state = secrets.token_urlsafe(32)

    st.session_state.chutes_pkce_verifier = verifier
    st.session_state.chutes_oauth_state = state
    st.session_state.auth_error = None
    store_oauth_state(state, verifier)

    params = {
        "response_type": "code",
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "scope": config["scope"],
        "state": state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{CHUTES_AUTH_BASE_URL}/idp/authorize?{urlencode(params)}"


def fetch_chutes_user_profile(access_token: str) -> dict | None:
    """Fetch the authenticated user's Chutes profile when the token permits it."""

    try:
        response = requests.get(
            f"{CHUTES_AUTH_BASE_URL}/idp/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=15,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        logger.exception("Unable to fetch Chutes user profile.")
        return None


def resolve_chutes_model(api_key: str, base_url: str) -> str | None:
    """
    Choose the Chutes model to use for chat completions.

    Prefer an explicit CHUTES_MODEL environment variable. If it is not set,
    query Chutes' OpenAI-compatible /models endpoint and use the first live
    model returned by the API.
    """

    configured_model = os.getenv("CHUTES_MODEL")
    if configured_model:
        return configured_model

    try:
        models_url = f"{base_url.rstrip('/')}/models"
        response = requests.get(
            models_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15,
        )
        response.raise_for_status()
        models_payload = response.json()
        live_models = models_payload.get("data", [])

        for model_info in live_models:
            model_id = model_info.get("id")
            if model_id:
                st.session_state["resolved_chutes_model"] = model_id
                return model_id
    except requests.RequestException:
        logger.exception("Unable to fetch live Chutes model list.")

    return None


def exchange_chutes_code_for_tokens(code: str, code_verifier: str) -> dict:
    """
    Exchange a Chutes authorization code for access and refresh tokens.

    The client secret and PKCE verifier are sent from the Streamlit backend only.
    """

    config = get_oauth_config()
    missing = [
        key
        for key in ("client_id", "client_secret", "redirect_uri")
        if not config[key]
    ]
    if missing:
        raise RuntimeError(
            "Missing Chutes OAuth configuration: " + ", ".join(missing)
        )

    response = requests.post(
        f"{CHUTES_AUTH_BASE_URL}/idp/token",
        data={
            "grant_type": "authorization_code",
            "client_id": config["client_id"],
            "client_secret": config["client_secret"],
            "code": code,
            "redirect_uri": config["redirect_uri"],
            "code_verifier": code_verifier,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def get_query_param(name: str) -> str | None:
    """Read a single URL query parameter from Streamlit's query-param mapping."""

    value = st.query_params.get(name)
    if isinstance(value, list):
        return value[0] if value else None
    return value


def handle_chutes_oauth_callback() -> None:
    """
    Complete Chutes OAuth when Streamlit is loaded with ?code=...&state=....

    On success, tokens are saved in st.session_state and the URL query params
    are cleared so refreshes do not attempt to reuse a single-use code.
    """

    error = get_query_param("error")
    code = get_query_param("code")
    returned_state = get_query_param("state")

    if error:
        st.session_state.auth_error = f"Chutes sign-in failed: {error}"
        st.query_params.clear()
        return

    if not code:
        return

    if not returned_state:
        st.session_state.auth_error = (
            "Missing Chutes OAuth state. Please start sign-in again."
        )
        st.query_params.clear()
        return

    expected_state = st.session_state.get("chutes_oauth_state")
    code_verifier = pop_oauth_code_verifier(returned_state)

    # Fallback for sessions where Streamlit preserved state across the redirect.
    if not code_verifier and expected_state == returned_state:
        code_verifier = st.session_state.get("chutes_pkce_verifier")

    if not code_verifier:
        st.session_state.auth_error = (
            "Chutes sign-in expired or could not be verified. Please try again."
        )
        st.query_params.clear()
        return

    try:
        token_response = exchange_chutes_code_for_tokens(code, code_verifier)
        access_token = token_response.get("access_token")
        if not access_token:
            raise RuntimeError("Chutes token response did not include access_token.")

        st.session_state.chutes_access_token = access_token
        st.session_state.chutes_refresh_token = token_response.get("refresh_token")
        st.session_state.chutes_token_response = token_response
        st.session_state.chutes_user_profile = fetch_chutes_user_profile(access_token)
        st.session_state.auth_error = None
    except requests.Timeout:
        st.session_state.auth_error = "Chutes token exchange timed out. Try again."
    except requests.HTTPError as error:
        st.session_state.auth_error = (
            "Chutes token exchange failed: "
            f"{error.response.status_code} {error.response.text}"
        )
    except Exception as error:
        logger.exception("Chutes OAuth callback failed.")
        st.session_state.auth_error = f"Chutes sign-in failed: {error}"
    finally:
        st.session_state.chutes_oauth_state = None
        st.session_state.chutes_pkce_verifier = None
        st.query_params.clear()
        st.rerun()


def is_chutes_authenticated() -> bool:
    """Return whether the current Streamlit session has a Chutes access token."""

    return bool(st.session_state.get("chutes_access_token"))


def sign_out_of_chutes() -> None:
    """Clear Chutes auth and document state from the Streamlit session."""

    for key in (
        "chutes_access_token",
        "chutes_refresh_token",
        "chutes_token_response",
        "chutes_user_profile",
        "chutes_oauth_state",
        "chutes_pkce_verifier",
    ):
        st.session_state[key] = None

    st.session_state.pdf_name = None
    st.session_state.pdf_text = ""
    st.session_state.pdf_chunks = []
    st.session_state.processing_error = None
    st.session_state.messages = []


# ---------------------------------------------------------------------------
# UI rendering helpers
# ---------------------------------------------------------------------------

def reset_document_state(uploaded_file_name: str) -> None:
    """Reset document-dependent state when the user uploads a different PDF."""

    st.session_state.pdf_name = uploaded_file_name
    st.session_state.pdf_text = ""
    st.session_state.pdf_chunks = []
    st.session_state.processing_error = None
    st.session_state.messages = []


def render_sidebar() -> None:
    """Render the corporate-looking sidebar."""

    with st.sidebar:
        st.title(APP_TITLE)
        st.caption("Secure document intelligence workspace")
        st.divider()

        if is_chutes_authenticated():
            profile = st.session_state.get("chutes_user_profile") or {}
            display_name = (
                profile.get("name")
                or profile.get("preferred_username")
                or profile.get("username")
                or profile.get("email")
                or "Chutes user"
            )
            st.success(f"Signed in as {display_name}")

            if st.button("Sign out", type="secondary"):
                sign_out_of_chutes()
                st.rerun()
        else:
            authorization_url = create_chutes_authorization_url()
            if authorization_url:
                st.link_button(
                    "Sign in with Chutes",
                    authorization_url,
                    type="primary",
                    use_container_width=True,
                )
            else:
                st.warning(st.session_state.auth_error)

        st.divider()
        st.caption("Sign in, upload a PDF, then ask questions about its contents.")


def render_document_status() -> None:
    """Show a concise status panel for the currently processed document."""

    if not st.session_state.pdf_name:
        return

    chunk_count = len(st.session_state.pdf_chunks)
    char_count = len(st.session_state.pdf_text)

    st.markdown(
        f"""
        <div class="status-panel">
            <strong>Document ready:</strong> {st.session_state.pdf_name}<br>
            <span>{char_count:,} extracted characters across {chunk_count:,} chunks.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat_history() -> None:
    """Replay the chat transcript stored in Streamlit session state."""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

handle_chutes_oauth_callback()
render_sidebar()

st.title(APP_TITLE)
st.markdown(
    '<p class="fortress-subtitle">Upload a PDF and ask focused questions about the document.</p>',
    unsafe_allow_html=True,
)

if st.session_state.auth_error:
    st.error(st.session_state.auth_error)

if not is_chutes_authenticated():
    st.info("Sign in with Chutes from the sidebar to upload and analyze a PDF.")
    st.stop()

uploaded_pdf = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    accept_multiple_files=False,
    help="FortressDoc AI currently supports text-based PDF files.",
)

if uploaded_pdf is not None:
    # A new file should start a fresh document session and chat transcript.
    if uploaded_pdf.name != st.session_state.pdf_name:
        reset_document_state(uploaded_pdf.name)

        with st.spinner("Extracting and chunking document text..."):
            try:
                extracted_text = extract_pdf_text(uploaded_pdf)

                if not extracted_text:
                    raise RuntimeError(
                        "No extractable text was found. This PDF may be scanned or image-only."
                    )

                st.session_state.pdf_text = extracted_text
                st.session_state.pdf_chunks = chunk_document_text(extracted_text)
            except Exception as error:
                st.session_state.processing_error = str(error)
                st.session_state.pdf_text = ""
                st.session_state.pdf_chunks = []

    if st.session_state.processing_error:
        st.error(st.session_state.processing_error)
    elif st.session_state.pdf_chunks:
        render_document_status()
else:
    # If the uploader is cleared, clear the document-specific state as well.
    if st.session_state.pdf_name is not None:
        st.session_state.pdf_name = None
        st.session_state.pdf_text = ""
        st.session_state.pdf_chunks = []
        st.session_state.processing_error = None
        st.session_state.messages = []

st.subheader("Document Chat")
render_chat_history()

chat_disabled = not bool(st.session_state.pdf_chunks)
placeholder_text = (
    "Ask a question about the uploaded document"
    if not chat_disabled
    else "Upload a text-based PDF to start chatting"
)

user_prompt = st.chat_input(placeholder_text, disabled=chat_disabled)

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    context = select_context_for_prompt(st.session_state.pdf_chunks, user_prompt)
    assistant_response = query_chutes_api(context=context, user_prompt=user_prompt)

    st.session_state.messages.append(
        {"role": "assistant", "content": assistant_response}
    )

    with st.chat_message("assistant"):
        st.markdown(assistant_response)
