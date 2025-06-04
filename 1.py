import os
import nltk
import ssl
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# SSL workaround and download resources
ssl._create_default_https_context = ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download('punkt')

# Intent patterns and responses
intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up"],
        "responses": ["Hello!", "Hey there!", "Hi! How can I assist you today?"]
    },
    {
        "tag": "courses",
        "patterns": ["AIPA", "AI", "Artificial Intelligence"],
        "responses": ["Here is the syllabus for AIPA: https://aimicrodegree.org/", "See you later!"]
    },
    {
        "tag": "Topic",
        "patterns": ["Computer Fundamentals", "CF", "Computer basics"],
        "responses": ["Please refer this : https://github.com/shail1806/LDA_NLP/blob/main/Computer%20Fundamentals.txt", "https://github.com/shail1806/LDA_NLP/blob/main/Computer%20Fundamentals.txt"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose"],
        "responses": ["I am a chatbot designed to help you.", "I can answer questions and assist you.", "Just your friendly digital assistant!"]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the issue?", "Tell me how I can assist you."]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age"],
        "responses": ["I don't have an age. I'm digital!", "I was just born in code form!", "Age is just data to me."]
    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today"],
        "responses": ["I can't check the weather in real-time, but it's always sunny in code!", "Try a weather app for real-time info."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": [
            "Start by tracking your income and expenses.",
            "Use the 50/30/20 rule: 50% needs, 30% wants, 20% savings.",
            "Create goals and align your spending accordingly."
        ]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": [
            "A credit score shows how creditworthy you are.",
            "You can check it using apps like Credit Karma.",
            "To improve it, pay your bills on time and reduce debt."
        ]
    }
]

# Vectorizer and classifier training
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

tags, patterns = [], []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

X = vectorizer.fit_transform(patterns)
y = tags
clf.fit(X, y)

# Chatbot response function
def chatbot_response(user_input):
    input_vec = vectorizer.transform([user_input])
    tag = clf.predict(input_vec)[0]
    for intent in intents:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Sorry, I didn't understand that."

# Main chat interface
def main():
    st.title("💬 Chatbot")

    # Initialize session state for conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input box
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and store bot response
        response = chatbot_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        # Optional goodbye handler
        if response.lower() in ['goodbye', 'bye', 'see you later', 'take care']:
            st.info("Chat ended. Refresh to restart.")

if __name__ == "__main__":
    main()
