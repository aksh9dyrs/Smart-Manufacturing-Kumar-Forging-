

# 🏭 Smart Manufacturing AI Dashboard

An AI-powered, full-stack smart manufacturing analytics dashboard combining vector search, event insights, and large language models (LLMs). It supports real-time data retrieval, similarity search using embeddings, interactive querying with LLMs, and autonomous AI agent assistance.

---

## 🚀 Overview

This project enables manufacturers to:

* Track events like **Breakdowns**, **Production**, and **Maintenance**
* Query and analyze using **LLMs (Gemini + LLaMA)**
* Visualize events using **UMAP**
* Ask intelligent questions and get summarized insights
* Evaluate AI-generated answers using **RAGAS metrics**
* Use a **multi-agent system** that mimics human-level support

---

## 🧠 Features

| Feature                             | Description                                                               |
| ----------------------------------- | ------------------------------------------------------------------------- |
| 🔍 **Event Search & Filtering**     | Retrieve, filter, and view all event types from PostgreSQL                |
| 📐 **Cosine Similarity Engine**     | Compute similarity between events using `pgvector` and cosine metrics     |
| 🧠 **Ask LLaMA/Gemini**             | Ask questions about the data using LLaMA 2 (local) or Gemini (cloud)      |
| 🌐 **Web Search Agent**             | Combine internal data with real-time web info using Google & Wikipedia    |
| 🤖 **AI Agent Assistant**           | Embedding + Prompt-based assistant for similarity, summaries, and RAG     |
| 📊 **UMAP Embedding Visualization** | Cluster event embeddings for pattern analysis                             |
| 📈 **RAGAS Evaluation**             | Evaluate QA system on metrics like faithfulness, context recall, and more |
| 🔐 **Secure API Key Handling**      | Uses `.env` for OpenAI keys and environment variables                     |

---

## 🧑‍💻 Tech Stack

| Layer             | Tech                                                                                         |
| ----------------- | -------------------------------------------------------------------------------------------- |
| **Frontend**      | [Streamlit](https://streamlit.io)                                                            |
| **Backend**       | [Flask](https://flask.palletsprojects.com/) for LLaMA API                                    |
| **Database**      | [PostgreSQL](https://www.postgresql.org/) + [pgvector](https://github.com/pgvector/pgvector) |
| **LLMs**          | Google Gemini (`google-generativeai`) and Meta LLaMA 2 (via API)                             |
| **AI Agent**      | Embedding-based reasoning, Gemini prompts, similarity explanation                            |
| **Evaluation**    | [RAGAS](https://github.com/explodinggradients/ragas) QA scoring                              |
| **Visualization** | `UMAP`, `matplotlib`                                                                         |
| **Search Tools**  | `wikipedia`, `googlesearch-python`, `BeautifulSoup`                                          |
| **Security**      | `.env` secrets management via `python-dotenv`                                                |

---

## 📂 Project Structure

```bash
.
├── app.py                # Flask API for LLaMA-based interaction
├── app2.py               # Main Streamlit app with Gemini + RAG support
├── ragas_evaluation.py   # Evaluation pipeline for LLM responses
├── requirements.txt      # Python dependencies
├── .env                  # (Create this file to store your API keys securely)
```

---

## 🛡️ OpenAI API Key Usage

This project uses OpenAI's API for:

* Fallback AI generation (if Gemini fails or is disabled)
* Custom RAG prompts and summaries

### 🔐 Secure Setup

**Step 1: Create a `.env` file**

```bash
touch .env
```

**Step 2: Add your API key to `.env`**

```env
OPENAI_API_KEY=sk-xxxxxxx
```

**Step 3: Load it securely in Python**

```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

> 🔒 Avoid hardcoding API keys directly in scripts (`app2.py`).

---

## 📦 Installation

### 1. Clone the Repo

```bash
git clone https://github.com/yourusername/smart-manufacturing-dashboard.git
cd smart-manufacturing-dashboard
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Set Up PostgreSQL

Ensure `pgvector` extension is installed, and `manufacturing_events` table is present.

### 4. Run the App

* For Streamlit Dashboard:

  ```bash
  streamlit run app2.py
  ```

* For LLaMA Flask API:

  ```bash
  python app.py
  ```

---

## 📄 `requirements.txt`

```txt
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

## 📊 RAG Evaluation (RAGAS)

Evaluate the system’s question-answering quality using:

* ✅ **Faithfulness**
* ✅ **Answer Relevancy**
* ✅ **Context Recall**
* ✅ **Context Precision**

Found in `ragas_evaluation.py`, where synthetic QA pairs are evaluated using your custom RAG pipeline.

---

## 📸 Screenshots

*Add screenshots of:*

* Dashboard home
* Similarity results
* AI-generated answers
* UMAP plots
* RAGAS scores

---

## 👤 Author

**Aksh Kumar**
AI Researcher | Manufacturing Data Engineer
📧 [aksh@example.com](mailto:aksh@example.com)
🔗 [LinkedIn](https://linkedin.com/in/yourname) • [GitHub](https://github.com/yourusername)

---

## 📌 Future Enhancements

* 📡 Deploy with Docker for production
* 🌐 Host online via Streamlit Cloud or Render
* 📈 Time-series forecasting of events
* 🔐 OAuth-based login for dashboard access


