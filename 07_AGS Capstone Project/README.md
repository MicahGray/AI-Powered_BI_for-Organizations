# 🔥 InsightForge – AI-Powered Business Intelligence Assistant

> Capstone project for the **Purdue Generative AI Specialist** program.

InsightForge is an enterprise-grade AI business intelligence assistant that combines LLMs, Retrieval-Augmented Generation (RAG), machine learning, and interactive visualizations to deliver actionable insights from sales data.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)

---

## ✨ Features

| Category | Details |
|----------|---------|
| **RAG System** | FAISS vector store + custom pandas-based retriever with semantic fallback |
| **Chain Prompts** | Multi-step LCEL pipeline: classify query → retrieve context → generate insight |
| **Conversation Memory** | Session-based sliding window (last 5 turns) for context-aware responses |
| **ML Models** | Customer segmentation (KMeans), anomaly detection (Isolation Forest), sales forecasting (Holt-Winters) |
| **LLM Evaluation** | LLM-based QA grading against 5 ground-truth question-answer pairs |
| **Visualizations** | 8 interactive Plotly charts (trends, forecasts, segmentation, anomalies, demographics) |
| **Streamlit UI** | 4-tab interface: Dashboard, Explorer, AI Assistant, Evaluation |

---

## 🛠️ Tech Stack

- **LLM:** OpenAI GPT-4o-mini via LangChain (LCEL)
- **RAG:** FAISS vector store + OpenAI Embeddings
- **ML:** scikit-learn, statsmodels
- **Visualization:** Plotly (dark-themed interactive charts)
- **UI Framework:** Streamlit
- **Data:** pandas, numpy

---

## 📁 Project Structure

```
├── Enabling_AI_Powered_Business_Intelligence.py  # Main application
├── Datasets/
│   └── sales_data.csv                            # Sales dataset (2022–2024)
├── requirements.txt                               # Python dependencies
├── .streamlit/
│   └── config.toml                                # Streamlit theme config
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API key

### Local Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/insightforge.git
cd insightforge

# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run the app
streamlit run Enabling_AI_Powered_Business_Intelligence.py
```

### Streamlit Community Cloud

This app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud).
Add your `OPENAI_API_KEY` in the app's **Secrets** settings (Settings → Secrets):
```toml
OPENAI_API_KEY = "sk-..."
```

---

## 📊 Application Tabs

### 1. Dashboard
KPI cards (total sales, average, satisfaction, anomalies) and 8 interactive charts covering sales trends, forecasts, product/region performance, customer segmentation, anomalies, demographics, and satisfaction.

### 2. Explorer
Interactive filters for product, region, and date range with dynamic chart and table updates.

### 3. AI Assistant
Chat interface powered by RAG. Ask questions about the data in natural language — the system classifies the query, retrieves relevant data, and generates data-driven insights with conversation memory.

### 4. Evaluation
Run LLM-based QA evaluation against 5 ground-truth pairs to assess the RAG system's accuracy.

---

## 📑 Capstone Requirements Mapping

| Step | Requirement | Implementation |
|------|-------------|----------------|
| 1 | Data Preparation | `load_data()` — CSV loading with derived columns |
| 2 | Knowledge Base | 5 structured documents (stats, time, products, regions, demographics) |
| 3 | LLM + Custom Retriever | Pattern-based pandas retriever + FAISS semantic fallback |
| 4 | Chain Prompts | LCEL: classify → retrieve → generate |
| 5 | RAG System | FAISS vector store with OpenAI embeddings |
| 6 | Memory | Session-state conversation memory (window = 5) |
| 7 | Evaluation | LLM QA grading with 5 ground-truth pairs |
| 8 | Visualization + UI | 8 Plotly charts + 4-tab Streamlit interface |

---

## 📜 License

This project was created as part of the Purdue University Generative AI Specialist program.
