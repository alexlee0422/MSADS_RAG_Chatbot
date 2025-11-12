# MSADS_RAG_Chatbot

## 🚀 Project Goal

This project is an **interactive RAG chatbot** designed to answer any question about the **University of Chicago's Master of Science in Applied Data Science (MSADS) program**.

The chatbot acts as an intelligent assistant, providing accurate and context-aware information to prospective students, current students, or faculty.

## Our Approach: A Fine-Tuned RAG System

To ensure high-quality answers, this chatbot is built on a custom Retrieval-Augmented Generation (RAG) pipeline with a uniquely fine-tuned embedding model.

Our methodology is broken into three main stages:

### 1. Data Collection & Preparation
First, we scraped the official UChicago MSADS program website to gather all relevant, up-to-date information. This raw text data serves as the foundational knowledge for our chatbot.

### 2. Fine-Tuning the Embedding Model
Standard embedding models are good, but not specialized. To improve retrieval accuracy, we fine-tuned our own model:
* **Q&A Pair Generation:** We used a Large Language Model (LLM) to automatically generate a high-volume, synthetic dataset of question-and-answer pairs based on the scraped website data.
* **Model Training:** This new Q&A dataset was then used to fine-tune a sentence-transformer (embedding) model. This training teaches the model to better understand the specific nuances and terminology of the MSADS program, making it far more effective at finding relevant answers.

### 3. RAG Implementation
The fine-tuned embedding model is the core of our RAG pipeline. When a user asks a question, the system retrieves the most relevant information from the scraped data (stored in a vector database) and feeds it to an LLM to generate a clear, accurate, and human-like response.

---

## ☁️ Deployment

The final application is built with a **Streamlit** user interface. This provides a simple, interactive chat window for users to ask questions.

The chatbot is deployed live and is publicly accessible via **Streamlit Community Cloud**, which integrates directly with the GitHub repository.
