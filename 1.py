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
    "patterns": [
        "AIPA", "AI", "Artificial Intelligence",
        "Can you show me the syllabus?", "What is the syllabus?",
        "Where can I find the syllabus?", "Syllabus for AI",
        "Give me the course outline", "AI course contents", "Syllabus of AIPA"
    ],
    "responses": [
        'Here is the syllabus for AIPA: https://www.youtube.com/embed/AHMEtNAZTP4?si=vmup-3QQereqg8IZ',
        "The full course outline is available at: https://aimicrodegree.org/",
        "You can check the syllabus here: https://aimicrodegree.org/syllabus"
    ]
    },

    {
        "tag": "Schedule",
        "patterns": ["Schedule of AIPA course", "Duration of AIPA course","Duration of course"],
        "responses": ["The schedule for AIPA is 1 year.", "The duration of AIPA course is 1 year", "The duration of AIPA course is 1 year"]#link
    },
    {
        "tag": "Topic-1",
        "patterns": ["Computer Fundamentals", "CF", "Computer basics","Basics of Computer","Computer Hardware and Software","Module-1","Lesson 1","Chapter 1"],
        "responses": ["Please refer this : https://github.com/shail1806/LDA_NLP/blob/main/Computer%20Fundamentals.txt", "https://github.com/shail1806/LDA_NLP/blob/main/Computer%20Fundamentals.txt"]#link
    },
    {
        "tag": "Topic-2",
        "patterns": ["Python", "Py","Python Programming","Module-2","Chapter 2","Lesson 2"],
        "responses": ["Please refer this : https://github.com/shail1806/LDA_NLP/blob/main/Python.txt", "https://github.com/shail1806/LDA_NLP/blob/main/Python.txt"]#link
    },
    {
        "tag": "Topic-3",
        "patterns": ["Database", "DBMS", "DB","Module-3","Chapter 3","Lesson 3"],
        "responses": ["Please refer this : https://github.com/shail1806/LDA_NLP/blob/main/Data%20Science.txt", "https://github.com/shail1806/LDA_NLP/blob/main/Data%20Science.txt"]#link
    },
    {
        "tag": "Topic-4",
        "patterns": ["Data Science", "DS","Module-4","SQL","Data Analysis","Chapter 4","Lesson 4"],
        "responses": ["Please refer this : https://github.com/shail1806/LDA_NLP/blob/main/Data%20Science.txt", "https://github.com/shail1806/LDA_NLP/blob/main/Data%20Science.txt"]#link
    },
    {
        "tag": "Topic-5",
        "patterns": ["Artificial Intelligence- Machine Learning", "AI","AI-ML","ML","Module-5","Chapter 5","Lesson 5"],
        "responses": ["Please refer this : https://github.com/shail1806/LDA_NLP/blob/main/AI.txt", "https://github.com/shail1806/LDA_NLP/blob/main/AI.txt"]#link
    },
    {
        "tag": "Topic-6",
        "patterns": ["Deep Learning", "DL","Module-6","Chapter 6","Lesson 6"],#link 
        "responses": ["Please refer this : https://github.com/shail1806/LDA_NLP/blob/main/Deep%20Learning.txt", "https://github.com/shail1806/LDA_NLP/blob/main/Deep%20Learning.txt"]
    },
    {
        "tag": "Topic-7",
        "patterns": ["Natural Language Processing", "NLP","NLP-ML","Module-7","Chapter 7","Lesson 7"],
        "responses": ["Please refer this : https://github.com/shail1806/LDA_NLP/blob/main/Cloud%20Computing.txt", "https://github.com/shail1806/LDA_NLP/blob/main/Cloud%20Computing.txt"]#link
    },
    {
        "tag": "about",
        "patterns": ["What can you do?", "Who are you?", "What is your purpose?"],
        "responses": ["I am a chatbot designed to help you.", "I can answer questions and assist you.", "Just your friendly digital assistant!","I can help you with AIPA course."]
    },
    {
        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the issue?", "Tell me how I can assist you."]
    },
    {
        "tag": "exam_info",
        "patterns": ["What kind of questions are on the test?", "What is the test like?", "How is the test structured?"],
        "responses": ["The test is a mix of multiple choice and true/false questions.", "It's a mix of coding and problem-solving.", "The test is a mix of coding and problem-solving."]
    },
    {
        "tag": "resources",
        "patterns": ["Where can i find the study materials?", "Where can I find resources are available?", "Can you provide study materials?", "Where can I find study materials?"],
        "responses": ["You can find study materials on Edunet website. And pdf notes on this github link:"] #link
    },
    {
        "tag": "learning tips",
        "patterns": ["How should I study AI?","Give me some study tips","How to learn better?","Tips to understand concepts","How do I prepare for AIPA?"],
        "responses": [
             "Start with small projects and revise key topics regularly.",
            "Use diagrams and real-world examples to understand AI concepts better."
        ]
    },
    {
        "tag": "contact_support",
        "patterns": ["I need to contact support", "Help me reach admin", "Where can I ask questions?", "Contact information", "Email support"],
        "responses": [
            "You can reach support via the contact form on https://aimicrodegree.org/contact",
            "For assistance, email: support@aimicrodegree.org"
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
    st.title("💬AIPA Chatbot")

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
