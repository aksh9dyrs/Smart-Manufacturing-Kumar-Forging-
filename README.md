# 🏠 Smart Manufacturing AI Dashboard(Kumar Forging Limited)

An AI-powered, full-stack smart manufacturing analytics dashboard combining vector search, event insights, and large language models (LLMs). It supports real-time data retrieval, similarity search using embeddings, interactive querying with LLMs, RAG evaluation, and autonomous AI agent assistance.

---

## 🚀 Overview

This project enables manufacturers to:

* Track events like **Breakdowns**, **Production**, and **Maintenance**
* Query and analyze data using **LLMs**: Google **Gemini 1.5 Flash** and Meta **LLaMA 2** (future scope)
* Visualize event embeddings using **UMAP**
* Get summarized insights using **RAG** + LLMs
* Evaluate AI-generated responses using **RAGAS**
* Interact with a **multi-agent system** that mimics human support

---

## 🧠 Features

| Feature                     | Description                                                               |
| --------------------------- | ------------------------------------------------------------------------- |
| 🔍 Event Search & Filtering | Retrieve, filter, and view all event types from PostgreSQL                |
| 📊 Cosine Similarity Engine | Compute similarity between events using pgvector and cosine metrics       |
| 🧠 Ask Gemini               | Ask questions about the data using Gemini (cloud LLM)                     |
| 🌐 Web Search Agent         | Combine internal data with Google & Wikipedia results                     |
| 🤖 AI Agent Assistant       | Embedding + Prompt-based assistant with summaries and insights            |
| 📊 UMAP Visualization       | Cluster event embeddings to discover patterns                             |
| 📈 RAGAS Evaluation         | Evaluate QA system on faithfulness, context recall, and relevancy metrics |
| 🔐 Secure API Key Handling  | Uses `.env` for environment-based key management                          |

---

## 🧱 Tech Stack

| Layer         | Tech                                                            |
| ------------- | --------------------------------------------------------------- |
| Frontend      | Streamlit                                                       |
| Backend       | (Handled within Streamlit; optionally extensible via API)       |
| Database      | PostgreSQL + pgvector                                           |
| LLMs          | Google Gemini 1.5 Flash (via `google-generativeai`)             |
| AI Agent      | Gemini + embedding similarity search + contextual QA generation |
| Evaluation    | RAGAS metrics (faithfulness, relevancy, recall, precision)      |
| Visualization | UMAP, matplotlib                                                |
| Search Tools  | Wikipedia API, Googlesearch Python, BeautifulSoup               |
| Security      | python-dotenv (.env based secrets loading)                      |

---

## 📂 Project Structure

```
.
├── app2.py               # Streamlit app with Gemini and RAG support
├── ragas_evaluation.py   # Evaluation pipeline for RAGAS metrics
├── test_ragas.py         # Test script for validating RAGAS installation
├── requirements.txt      # Python dependencies
├── .env                  # Store your API keys securely (not committed)
```

---

## 🔒 API Key Handling

Use `.env` for secure API key management:

1. Create a `.env` file:

```bash
touch .env
```

2. Add your keys:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxx
GEMINI_API_KEY=your-gemini-key
```

3. Load securely:

```python
from dotenv import load_dotenv
import os
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

**Avoid hardcoding keys in `app2.py`.**

---

## 📦 Installation

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/smart-manufacturing-dashboard.git
cd smart-manufacturing-dashboard
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up PostgreSQL

Ensure you have `pgvector` extension and the `manufacturing_events` table.

### 4. Run Application

**Streamlit App:**

```bash
streamlit run app2.py
```

---

## 📊 Requirements

```
streamlit>=1.24.0
pandas>=1.5.0
psycopg2-binary>=2.9.5
numpy>=1.21.0
matplotlib>=3.5.0
umap-learn>=0.5.3
google-generativeai>=0.3.0
ragas>=0.0.16
datasets>=2.12.0
torch>=2.0.0
transformers>=4.30.0
sentence-transformers>=2.2.2
python-dotenv>=1.0.0
```

---

## 📊 RAG Evaluation with RAGAS

Run the `ragas_evaluation.py` script to:

* Generate synthetic QA pairs from your DB
* Use Gemini to answer questions
* Score responses using:

  * ✅ Faithfulness
  * ✅ Answer Relevancy
  * ✅ Context Recall
  * ✅ Context Precision

---

## 🖼️ Screenshots

Include screenshots of:

* Dashboard Home View
* Event Similarity Results
* AI-generated Answers
* UMAP Cluster Plots
* RAG Evaluation Metrics

---

## 👤 Author

**Aksh Kumar**
AI Researcher | Manufacturing Data Engineer
📧 [aksh@example.com](mailto:aksh@example.com)
[LinkedIn](https://linkedin.com/in/your-profile) | [GitHub](https://github.com/yourusername)

---

## 📌 Future Enhancements

* 📡 Deploy with Docker
* 🌐 Host via Streamlit Cloud or Render
* 📊 Add Time-series Forecasting
* 🔐 Add OAuth Login for Secure Access

---

> Built with passion for smart factories and real-time AI-driven insights.
