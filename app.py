#---------------------------------------------------------------------------------------------#
import os
import json
import warnings
import streamlit as st
import numpy as np
import faiss
from groq import Groq
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
#---------------------------------------------------------------------------------------------#

load_dotenv("env")
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_MODEL   = "llama-3.3-70b-versatile"

BASE          = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH    = os.path.join(BASE, "rag_vector_store", "faiss_index.bin")
METADATA_PATH = os.path.join(BASE, "rag_vector_store", "chunks_metadata.json")

#---------------------------------------------------------------------------------------------#
@st.cache_resource(show_spinner=False)
def load_rag_resources():
    faiss_index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    model  = SentenceTransformer("all-MiniLM-L6-v2")
    client = Groq(api_key=GROQ_API_KEY)
    return faiss_index, records, model, client

#---------------------------------------------------------------------------------------------#
def retrieve_chunks(question, faiss_index, records, model, k=3):
    query_vec = model.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")
    scores, found_ids = faiss_index.search(query_vec, k)
    results = []
    for score, chunk_id in zip(scores[0], found_ids[0]):
        if chunk_id == -1:
            continue
        rec = dict(records[chunk_id])
        rec["score"] = float(score)
        results.append(rec)
    return results

def build_prompt(question, chunks):
    context_block = "\n\n---\n\n".join(
        f"[Result {i+1} | Category: {r['category']} | Intent: {r['intent']}]\n{r['text']}"
        for i, r in enumerate(chunks)
    )
    return f"""You are a warm, professional customer support assistant for an e-commerce company.

STRICT RULES:
1. Answer ONLY using the context provided below. Do NOT use any outside knowledge.
2. If the context does not contain a relevant answer, say:
   "I'm sorry, I don't have specific information about that. Please contact our support team directly."
3. If the question is off-topic (not about orders, accounts, payments, refunds, delivery, invoices, cancellations), politely say you can only help with customer support topics.
4. Write in clear, natural, human-readable language. Be warm but concise.
5. Never mention "the context", "the data", or "the chunks" in your reply.

CONTEXT:
{context_block}

CUSTOMER MESSAGE:
{question}

YOUR RESPONSE:"""

def get_answer(question, faiss_index, records, model, groq_client):
    greetings = ["hi", "hello", "hey", "salam", "assalam", "good morning",
                 "good evening", "good afternoon", "helo", "hii"]
    q_lower = question.strip().lower()

    if any(q_lower.startswith(g) for g in greetings) and len(q_lower.split()) <= 5:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content":
                f"""You are a warm, friendly customer support assistant named Zara.
The customer said: "{question}"
Respond warmly, introduce yourself as Zara, and invite them to ask about:
orders, cancellations, refunds, payments, account, delivery, or invoices.
Keep it to 2-3 sentences. No bullet points."""}],
            temperature=0.5, max_tokens=150,
        )
        return response.choices[0].message.content.strip()

    chunks = retrieve_chunks(question, faiss_index, records, model)
    if not chunks:
        return "I'm sorry, I couldn't find relevant information. Please contact our support team directly."

    prompt   = build_prompt(question, chunks)
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content":
                "You are a professional and empathetic customer support agent named Zara. "
                "Answer only from the provided context. Be human, warm, and clear."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2, max_tokens=512,
    )
    return response.choices[0].message.content.strip()

#---------------------------------------------------------------------------------------------#
def chat_bot_ui(faiss_index, records, embed_model, groq_client):

    quick_questions = [
        "How do I cancel my order?",
        "How can I get a refund?",
        "How do I track my delivery?",
        "How do I update my account?",
        "What payment methods are accepted?",
        "How do I check my invoice?",
    ]

    if "messages"         not in st.session_state: st.session_state.messages         = []
    if "pending_question" not in st.session_state: st.session_state.pending_question = None
    if "input_key"        not in st.session_state: st.session_state.input_key        = 0

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&display=swap');

    #MainMenu, footer, header { visibility: hidden; }

    .stApp { background: #f0f2f6; font-family: 'DM Sans', sans-serif; }
    /* Sidebar boundary */
    section[data-testid="stSidebar"] {
        border-right: 2px solid #d6d9de;}
    /* white chat card */
    .z-shell {
        background: #ffffff;
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 4px 24px rgba(0,0,0,0.08);
        max-width: 860px;
        margin: 0 auto;
    }

    /* header */
    .z-header {
        display: flex; align-items: center; gap: 12px;
        padding: 14px 18px;
        background: #ffffff;
        border-bottom: 1px solid #ebebeb;
    }
    .z-av {
        width: 40px; height: 40px; background: #4a90d9; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 15px; font-weight: 700; color: #fff; flex-shrink: 0;
    }
    .z-title { font-size: 14px; font-weight: 600; color: #1a1a1a; line-height: 1.2; }
    .z-sub   { font-size: 11px; color: #388e3c; display: flex; align-items: center; gap: 4px; margin-top: 2px; }
    .z-dot   { width: 6px; height: 6px; background: #388e3c; border-radius: 50%; display: inline-block; }

    /* chat body — same grey as sidebar/page */
    .z-chat {
        background: #f0f2f6;
        padding: 14px 14px 8px 14px;
        display: flex; flex-direction: column; gap: 10px;
        min-height: 120px;
    }

    /* welcome */
    .z-welcome { text-align: center; padding: 8px 20px 4px 20px; }
    .z-welcome-icon  { font-size: 34px; margin-bottom: 6px; }
    .z-welcome-title { font-size: 15px; font-weight: 600; color: #333; margin-bottom: 2px; }
    .z-welcome-sub   { font-size: 12px; color: #888; }

    /* rows */
    .z-row-user { display: flex; justify-content: flex-end; align-items: flex-end; gap: 7px; animation: zFade .2s ease; }
    .z-row-bot  { display: flex; justify-content: flex-start; align-items: flex-end; gap: 7px; animation: zFade .2s ease; }
    @keyframes zFade { from { opacity:0; transform:translateY(5px); } to { opacity:1; transform:translateY(0); } }

    /* avatars */
    .z-mav-z { width:26px; height:26px; background:#4a90d9; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:10px; font-weight:700; color:#fff; flex-shrink:0; }
    .z-mav-u { width:26px; height:26px; background:#388e3c; border-radius:50%; display:flex; align-items:center; justify-content:center; font-size:10px; font-weight:700; color:#fff; flex-shrink:0; }

    /* bubbles */
    .z-bub-user {
        background: #4a90d9; color: #fff;
        padding: 9px 14px; border-radius: 18px 18px 4px 18px;
        font-size: 13.5px; max-width: 70%; line-height: 1.5;
        font-family: 'DM Sans', sans-serif;
    }
    .z-bub-bot {
        background: #ffffff; color: #1a1a1a;
        padding: 9px 14px; border-radius: 18px 18px 18px 4px;
        font-size: 13.5px; max-width: 70%; line-height: 1.5;
        font-family: 'DM Sans', sans-serif;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    /* quick pills — same grey bg, no gap to input */
    .z-qwrap { background: #f0f2f6; padding: 8px 14px 0 14px; }
    .z-qwrap .stButton > button {
        background: #ffffff !important; color: #4a90d9 !important;
        border: 1px solid #c8d9f0 !important; border-radius: 16px !important;
        font-size: 12px !important; font-family: 'DM Sans', sans-serif !important;
        padding: 5px 13px !important; box-shadow: none !important;
        white-space: nowrap !important; transition: all .15s ease !important;
        height: auto !important; min-height: 0 !important; line-height: 1.4 !important;
    }
    .z-qwrap .stButton > button:hover {
        background: #4a90d9 !important; color: #fff !important; border-color: #4a90d9 !important;
    }

    /* input row */
    .z-input-wrap {
        background: #f0f2f6;
        padding: 8px 14px 14px 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* force outer container border (IMPORTANT) */
    div[data-testid="stTextInput"] > div {
        border: 2px solid #4a90d9 !important;
        border-radius: 22px !important;
        background: #ffffff !important;
        box-shadow: none !important;
    }

    /* force Streamlit internal focus wrapper */
    div[data-testid="stTextInput"]:focus-within > div {
        border: 2px solid #4a90d9 !important;
    }

    /* input itself */
    div[data-testid="stTextInput"] input {
        background: transparent !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;

        font-size: 13.5px !important;
        font-family: 'DM Sans', sans-serif !important;
        color: #1a1a1a !important;

        padding: 12px 14px !important;
    }

    /* remove focus border changes */
    div[data-testid="stTextInput"] input:focus {
        border: none !important;
        box-shadow: none !important;
    }

    /* hide label */
    div[data-testid="stTextInput"] label {
        display: none !important;
    }

    /* send */
    .z-send .stButton > button {
        background: #4a90d9 !important; color: #fff !important;
        border: none !important; border-radius: 22px !important;
        font-size: 13px !important; font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        width: auto !important; height: 40px !important;
        padding: 0 16px !important; box-shadow: 0 2px 8px rgba(74,144,217,0.35) !important;
        transition: all .15s ease !important; white-space: nowrap !important;
    }
    .z-send .stButton > button:hover { background: #357abd !important; transform: scale(1.04) !important; }

    /* clear */
    .z-clear .stButton > button {
        background: transparent !important; color: #aaa !important;
        border: 1px solid #ddd !important; border-radius: 22px !important;
        font-size: 13px !important; font-family: 'DM Sans', sans-serif !important;
        width: auto !important; height: 40px !important;
        padding: 0 14px !important; box-shadow: none !important;
        transition: color .15s !important; white-space: nowrap !important;
    }
    .z-clear .stButton > button:hover { color: #e53935 !important; border-color: #e53935 !important; }

    .block-container { padding-top: 0.5rem !important; }
    div[data-testid="stVerticalBlock"] > div { gap: 0rem; }
    </style>
    """, unsafe_allow_html=True)

    is_first_chat = len(st.session_state.messages) == 0

    # ── HEADER ──
    st.markdown("""
    <div class="z-shell">
    <div class="z-header">
        <div class="z-av">Z</div>
        <div>
            <div class="z-title">ZARA &mdash; Customer Support</div>
            <div class="z-sub"><span class="z-dot"></span> Online &nbsp;·&nbsp; Powered by RAG</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CHAT MESSAGES ──
    if is_first_chat:
        st.markdown("""
        <div class="z-chat">
            <div class="z-welcome">
                <div class="z-welcome-icon">🛍️</div>
                <div class="z-welcome-title">Hi, I'm Zara!</div>
                <div class="z-welcome-sub">Your e-commerce support assistant.<br>Ask me anything about your orders.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        bubbles = ""
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                bubbles += f"""
                <div class="z-row-user">
                    <div class="z-bub-user">{msg["content"]}</div>
                    <div class="z-mav-u">U</div>
                </div>"""
            else:
                bubbles += f"""
                <div class="z-row-bot">
                    <div class="z-mav-z">Z</div>
                    <div class="z-bub-bot">{msg["content"]}</div>
                </div>"""
        st.markdown(f'<div class="z-chat">{bubbles}</div>', unsafe_allow_html=True)

    # ── QUICK PILLS (only before first message) ──
    if is_first_chat:
        st.markdown('<div class="z-qwrap">', unsafe_allow_html=True)
        cols_q = st.columns(3)
        for i, q in enumerate(quick_questions):
            with cols_q[i % 3]:
                if st.button(q, key=f"quick_{i}"):
                    st.session_state.pending_question = q
        st.markdown('</div>', unsafe_allow_html=True)

    # ── INPUT BAR ──
    # Layout: [text input] [🗑 Clear Chat] [➤ Send (Enter)]
    st.markdown('<div class="z-input-wrap">', unsafe_allow_html=True)
    col_in, col_send, col_clear = st.columns([7, 1, 1])

    with col_in:
        user_input = st.text_input(
            label="msg",
            placeholder="Message Zara...",
            key=f"chat_input_{st.session_state.input_key}",
            label_visibility="hidden"
        )
    with col_send:
        st.markdown('<div class="z-send">', unsafe_allow_html=True)
        send_clicked = st.button("➤ Send", key="send_btn")
        st.markdown('</div>', unsafe_allow_html=True)
    with col_clear:
        st.markdown('<div class="z-clear">', unsafe_allow_html=True)
        clear_clicked = st.button("🗑 Clear", key="clear_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Enter key support — triggers the Send (➤) button on Enter keypress
    st.markdown("""
    <script>
    (function() {
        function hook() {
            var inputs = window.parent.document.querySelectorAll('input[type="text"]');
            inputs.forEach(function(inp) {
                if (inp._zHooked) return;
                inp._zHooked = true;
                inp.addEventListener('keydown', function(e) {
                    if (e.key !== 'Enter') return;
                    // Clicks the Send (➤) button — same as pressing Enter
                    var btns = window.parent.document.querySelectorAll('button');
                    for (var b of btns) {
                        if (b.innerText && b.innerText.trim().includes('➤')) { b.click(); break; }
                    }
                });
            });
        }
        hook(); setTimeout(hook, 600); setTimeout(hook, 1500);
    })();
    </script>
    """, unsafe_allow_html=True)

    # ── LOGIC ──
    def process_question(question):
        st.session_state.messages.append({"role": "user", "content": question})
        st.session_state.input_key += 1
        with st.spinner("Zara is typing..."):
            answer = get_answer(question, faiss_index, records, embed_model, groq_client)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.pending_question:
        q = st.session_state.pending_question
        st.session_state.pending_question = None
        process_question(q)

    if send_clicked and user_input and user_input.strip():
        process_question(user_input.strip())

    if clear_clicked:
        st.session_state.messages   = []
        st.session_state.input_key += 1
        st.rerun()

#---------------------------------------------------------------------------------------------#
def theory_subject_page(faiss_index, records, embed_model, groq_client):
    st.markdown("""
    <style>
    .stMarkdown>p { text-align: justify; font-size: 15px; }
    </style>
    """, unsafe_allow_html=True)
    chat_bot_ui(faiss_index, records, embed_model, groq_client)
    st.sidebar.image("logo.png", use_container_width=True)
    st.sidebar.title("About US:")
    st.sidebar.markdown("""
        - **1:** Fatima Shahid   (FA23-BBD-042)
        - **2:** Hifza Khizer    (FA23-BBD-053)
        - **3:** Ifrah Jamil     (FA23-BBD-059)
        - **4:** Faizan Akram    (FA23-BBD-090)
    """)

#---------------------------------------------------------------------------------------------#
def main():
    st.set_page_config(
        page_title="Chat Zara",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown("""
        <h1 style="text-align:center; color:#1a1a1a; font-weight:600;
                   font-size:26px; font-family:'DM Sans',sans-serif;
                   padding-top:10px; margin-bottom:0;">
            NLP For Business &mdash; Lab Final Project
        </h1>
    """, unsafe_allow_html=True)

    with st.spinner("Loading Zara..."):
        faiss_index, records, embed_model, groq_client = load_rag_resources()

    theory_subject_page(faiss_index, records, embed_model, groq_client)

if __name__ == "__main__":
    main()
#---------------------------------------------------------------------------------------------#
