
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(
    page_title="The AI Chatbot",
    page_icon="🤖",
    layout="centered"
)

# ---------------------------
# Load Model
# ---------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ---------------------------
# Knowledge Base
# ---------------------------
knowledge_base = {
    "What is your name?": "🤖 I am a smart AI chatbot built with Python and Streamlit.",
    "How are you?": "😃 I am doing great, thank you! How about you?",
    "What is AI?": "💡 Artificial Intelligence is the simulation of human intelligence in machines.",
    "Tell me a joke": "😂 Why did the computer show up at work late? Because it caught a virus!"
}

questions = list(knowledge_base.keys())
answers = list(knowledge_base.values())
question_embeddings = model.encode(questions, convert_to_tensor=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.title("⚙️ About this Chatbot")
    st.write("Built with:")
    st.markdown("- Python 🐍")
    st.markdown("- Streamlit 🎨")
    st.markdown("- Sentence Transformers 🤖")
    st.markdown("---")
    st.caption("👨‍💻 Developed by **Arpinderjit Singh**")

# ---------------------------
# Main Chat UI
# ---------------------------
st.title("🤖 The AI Chatbot")
st.markdown("### Ask me anything below ⬇️")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input (chat-style)
user_input = st.chat_input("Type your question...")

if user_input:
    # Save user message
    st.session_state.history.append({"role": "user", "content": user_input})

    # Generate bot response
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    similarity_scores = util.cos_sim(user_embedding, question_embeddings)
    best_match_idx = similarity_scores.argmax().item()
    response = answers[best_match_idx]

    # Save bot message
    st.session_state.history.append({"role": "assistant", "content": response})

# Display all messages
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])
