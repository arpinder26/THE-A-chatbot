import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load embedding model (small + fast)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Knowledge base (expand as needed)
knowledge_base = {
    "What is your name?": "I am a simple AI chatbot built with Python and Streamlit.",
    "How are you?": "I am doing great, thank you! How about you?",
    "What is AI?": "Artificial Intelligence is the simulation of human intelligence in machines.",
    "Tell me a joke": "Why did the computer show up at work late? Because it caught a virus!"
}

# Precompute embeddings for knowledge base
questions = list(knowledge_base.keys())
answers = list(knowledge_base.values())
question_embeddings = model.encode(questions, convert_to_tensor=True)

# Streamlit UI
st.title("🤖 Smarter AI Chatbot (v2)")
st.write("Now powered by **Sentence Transformers** for better understanding!")

user_input = st.text_input("Ask me anything:")

if st.button("Ask"):
    if user_input:
        # Encode user input
        user_embedding = model.encode(user_input, convert_to_tensor=True)
        
        # Compute cosine similarity
        similarity_scores = util.cos_sim(user_embedding, question_embeddings)
        
        # Get best match
        best_match_idx = similarity_scores.argmax().item()
        response = answers[best_match_idx]
        
        st.markdown(f"**Chatbot:** {response}")
    else:
        st.warning("Please type a question first.")

