import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# === CONFIGURATION ===
CHROMA_DB_DIR = '../../chroma_db'
COLLECTION_NAME = 'government_schemes_chunks'
TOP_K = 15
GEMINI_MODEL = "models/gemini-2.0-flash"
API_KEY = "AIzaSyC-K1zRmfqjaW8vXsI6lSIODL2zZObFq14"

CORE_FIELDS = {"Description"}
LONG_FIELDS = {"Application Process", "Eligibility Criteria", "Documents Required", "Benefits"}

# === Initialize Clients ===
@st.cache_resource
def init_clients():
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_collection(COLLECTION_NAME)
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    genai.configure(api_key=API_KEY)
    gemini = genai.GenerativeModel(GEMINI_MODEL)
    return collection, embed_model, gemini

collection, embed_model, gemini = init_clients()

# === FUNCTIONS ===
def detect_relevant_field(query):
    query = query.lower()
    if any(word in query for word in ["apply", "application", "register", "procedure", "how to avail"]):
        return "Application Process"
    elif "benefit" in query or "advantages" in query:
        return "Benefits"
    elif "eligibility" in query or "eligible" in query or "who can apply" in query:
        return "Eligibility Criteria"
    elif "document" in query or "paperwork" in query:
        return "Documents Required"
    else:
        return None

def retrieve_context(user_query):
    relevant_field = detect_relevant_field(user_query)
    query_embedding = embed_model.encode([user_query])

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K
    )

    docs = results['documents'][0]
    metas = results['metadatas'][0]

    scheme_order = []
    for meta in metas:
        pid = meta['parent_doc_id']
        if pid not in scheme_order:
            scheme_order.append(pid)
        if len(scheme_order) == 2:
            break

    final_context = []
    for idx, pid in enumerate(scheme_order):
        scheme_data = collection.get(where={"parent_doc_id": pid})
        scheme_name = ""
        section = []

        for doc, meta in zip(scheme_data['documents'], scheme_data['metadatas']):
            field = meta['field']
            if field == "Scheme Name" and not scheme_name:
                scheme_name = doc.split(":",1)[1].strip()
                section.append(f"Scheme Name: {scheme_name}")
            if field in CORE_FIELDS:
                section.append(f"{field}: {doc.split(':',1)[1].strip()}")

        if idx == 0 and relevant_field:
            for doc, meta in zip(scheme_data['documents'], scheme_data['metadatas']):
                if meta['field'] == relevant_field:
                    section.append(f"{relevant_field}: {doc.split(':',1)[1].strip()}")

        final_context.append("\n".join(section))

    return "\n\n---\n\n".join(final_context)

def generate_answer(query, context):
    prompt = f"""
You are an intelligent assistant for answering questions about Indian government schemes.

Here is the relevant information retrieved from the database:

{context}

Based on this, answer the following question clearly and concisely:

"{query}"
"""
    response = gemini.generate_content(prompt)
    return response.text

# === STREAMLIT UI ===
st.set_page_config(page_title="IndiaGov Schemes Assistant", layout="wide")
st.title("🇮🇳 Government Schemes Assistant")
st.write("Ask about schemes, eligibility, benefits, or how to apply – chat style!")

# Session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input box
user_input = st.chat_input("Ask your question about Indian government schemes...")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle new user message
if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = retrieve_context(user_input)
            answer = generate_answer(user_input, context)
            st.markdown(answer)

        # Store assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})

        with st.expander("🔍 View Retrieved Context (For Debugging)"):
            st.text(context)
