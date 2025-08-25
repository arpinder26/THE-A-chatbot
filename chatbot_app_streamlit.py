import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.set_page_config(page_title="Simple Chatbot", page_icon="🤖")

st.title("🤖 My Simple Chatbot")
st.write("Type a question below and I'll try to answer!")

# Knowledge base: Q&A pairs
qa_pairs = [
    ("hi", "Hello! How can I help you today?"),
    ("hello", "Hi there! What can I do for you?"),
    ("hey", "Hey! Need any help?"),
    ("what is your name", "I'm SimpleBot, your friendly chatbot 🤖"),
    ("who made you", "I was created as a simple demo in Python + Streamlit."),
    ("what can you do", "I can answer basic questions you teach me via Q&A pairs."),
    ("what are your hours", "We're open 9am–5pm, Monday to Friday."),
    ("when are you open", "We're open 9am–5pm, Monday to Friday."),
    ("how much does it cost", "Basic plan is free, Pro is $10/month."),
    ("what is the price", "Basic plan is free, Pro is $10/month."),
    ("do you offer support", "Yes! Basic email support for all users."),
    ("where are you located", "We're fully online 🌐"),
    ("bye", "Goodbye! Have a great day! 👋"),
]

# Build TF-IDF model
questions = [q for q, a in qa_pairs]
answers = [a for q, a in qa_pairs]
vectorizer = TfidfVectorizer().fit(questions)
question_matrix = vectorizer.transform(questions)

def get_response(user_input, threshold=0.2):
    vec = vectorizer.transform([user_input])
    sims = cosine_similarity(vec, question_matrix)[0]
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])

    if best_score < threshold:
        return "I didn't quite get that. Can you rephrase?"
    return answers[best_idx]

# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

if prompt := st.chat_input("Ask me something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").markdown(prompt)

    response = get_response(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
