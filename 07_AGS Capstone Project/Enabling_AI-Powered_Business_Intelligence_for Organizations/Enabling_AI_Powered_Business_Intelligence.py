# %% [markdown]
# # InsightForge – Enabling AI‑Powered Business Intelligence for Organizations
#
# AI‑powered Business Intelligence Assistant using LLMs, RAG, forecasting,
# anomaly detection, customer segmentation, Streamlit UI, and LLM evaluation.
#
# Capstone project for Purdue Generative AI Specialist program.

# %%
# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os, re, textwrap

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from statsmodels.tsa.holtwinters import ExponentialSmoothing

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

# ============================================================================
# STREAMLIT PAGE CONFIG  (must be first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="InsightForge – AI Business Intelligence",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──── Custom CSS for premium look ────
st.markdown("""
<style>
    /* Main background & text */
    .stApp {background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%);}
    h1, h2, h3, h4 {color: #e2e8f0 !important;}
    p, li, span, label {color: #cbd5e1 !important;}

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] {color: #38bdf8 !important; font-weight: 700;}
    [data-testid="stMetricLabel"] {color: #94a3b8 !important;}

    /* Tabs */
    button[data-baseweb="tab"] {color: #94a3b8 !important;}
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15,12,41,0.95);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# STEP 1 – DATA PREPARATION
# ============================================================================
@st.cache_data
def load_data():
    """Load and prepare the sales dataset."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "Datasets", "sales_data.csv")

    if not os.path.exists(csv_path):
        st.error(f"Dataset not found at: {csv_path}")
        st.stop()

    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Year"] = df["Date"].dt.year
    df["Quarter"] = df["Date"].dt.to_period("Q").astype(str)
    return df

df = load_data()


# ============================================================================
# STEP 2 – KNOWLEDGE BASE CREATION
# ============================================================================
@st.cache_data
def generate_advanced_data_summary(_df):
    """
    Generate comprehensive analysis of business data.
    Returns both a summary dict and formatted text for the knowledge base.
    """
    summary = {}

    # ── Statistical measures ──
    summary["total_sales"] = _df["Sales"].sum()
    summary["average_sales"] = _df["Sales"].mean()
    summary["median_sales"] = _df["Sales"].median()
    summary["std_sales"] = _df["Sales"].std()
    summary["min_sales"] = _df["Sales"].min()
    summary["max_sales"] = _df["Sales"].max()

    # ── Time-based analysis ──
    summary["sales_by_month"] = _df.groupby("Month")["Sales"].sum()
    summary["sales_by_quarter"] = _df.groupby("Quarter")["Sales"].sum()
    summary["sales_by_year"] = _df.groupby("Year")["Sales"].sum()

    # ── Product analysis ──
    summary["sales_by_product"] = _df.groupby("Product")["Sales"].sum()
    summary["avg_satisfaction_by_product"] = _df.groupby("Product")["Customer_Satisfaction"].mean()

    # ── Region analysis ──
    summary["sales_by_region"] = _df.groupby("Region")["Sales"].sum()
    summary["avg_satisfaction_by_region"] = _df.groupby("Region")["Customer_Satisfaction"].mean()

    # ── Customer segmentation ──
    age_bins = [0, 25, 35, 45, 55, 65, 100]
    age_labels = ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
    _df_copy = _df.copy()
    _df_copy["Age_Group"] = pd.cut(_df_copy["Customer_Age"], bins=age_bins, labels=age_labels)
    summary["sales_by_age_group"] = _df_copy.groupby("Age_Group", observed=False)["Sales"].sum()
    summary["sales_by_gender"] = _df.groupby("Customer_Gender")["Sales"].sum()

    return summary


@st.cache_data
def build_knowledge_base_documents(_df, _summary):
    """Create structured text documents for the RAG vector store."""
    documents = []

    # Document 1: Overall statistics
    overall_doc = textwrap.dedent(f"""\
        OVERALL SALES STATISTICS
        Total Sales: ${_summary['total_sales']:,.2f}
        Average Sale: ${_summary['average_sales']:,.2f}
        Median Sale: ${_summary['median_sales']:,.2f}
        Standard Deviation: ${_summary['std_sales']:,.2f}
        Min Sale: ${_summary['min_sales']:,.2f}
        Max Sale: ${_summary['max_sales']:,.2f}
        Total Records: {len(_df)}
        Date Range: {_df['Date'].min().strftime('%Y-%m-%d')} to {_df['Date'].max().strftime('%Y-%m-%d')}
    """)
    documents.append(Document(page_content=overall_doc, metadata={"topic": "overall_statistics"}))

    # Document 2: Sales by time period
    time_doc = "SALES BY TIME PERIOD\n\n"
    time_doc += "Monthly Sales:\n" + _summary["sales_by_month"].to_string() + "\n\n"
    time_doc += "Quarterly Sales:\n" + _summary["sales_by_quarter"].to_string() + "\n\n"
    time_doc += "Yearly Sales:\n" + _summary["sales_by_year"].to_string()
    documents.append(Document(page_content=time_doc, metadata={"topic": "time_analysis"}))

    # Document 3: Product performance
    product_doc = "PRODUCT PERFORMANCE ANALYSIS\n\n"
    product_doc += "Total Sales by Product:\n" + _summary["sales_by_product"].to_string() + "\n\n"
    product_doc += "Average Customer Satisfaction by Product:\n"
    product_doc += _summary["avg_satisfaction_by_product"].to_string()
    best_product = _summary["sales_by_product"].idxmax()
    product_doc += f"\n\nBest Performing Product: {best_product} with ${_summary['sales_by_product'][best_product]:,.2f} in total sales"
    documents.append(Document(page_content=product_doc, metadata={"topic": "product_analysis"}))

    # Document 4: Regional performance
    region_doc = "REGIONAL PERFORMANCE ANALYSIS\n\n"
    region_doc += "Total Sales by Region:\n" + _summary["sales_by_region"].to_string() + "\n\n"
    region_doc += "Average Customer Satisfaction by Region:\n"
    region_doc += _summary["avg_satisfaction_by_region"].to_string()
    best_region = _summary["sales_by_region"].idxmax()
    region_doc += f"\n\nTop Performing Region: {best_region} with ${_summary['sales_by_region'][best_region]:,.2f} in total sales"
    documents.append(Document(page_content=region_doc, metadata={"topic": "regional_analysis"}))

    # Document 5: Customer demographics
    demo_doc = "CUSTOMER DEMOGRAPHICS AND SEGMENTATION\n\n"
    demo_doc += "Sales by Age Group:\n" + _summary["sales_by_age_group"].to_string() + "\n\n"
    demo_doc += "Sales by Gender:\n" + _summary["sales_by_gender"].to_string() + "\n\n"
    demo_doc += f"Average Customer Age: {_df['Customer_Age'].mean():.1f}\n"
    demo_doc += f"Average Customer Satisfaction: {_df['Customer_Satisfaction'].mean():.2f}"
    documents.append(Document(page_content=demo_doc, metadata={"topic": "customer_demographics"}))

    return documents


summary = generate_advanced_data_summary(df)
kb_documents = build_knowledge_base_documents(df, summary)


# ============================================================================
# STEP 3 – ML MODELS (Segmentation, Anomaly, Forecast)
# ============================================================================
@st.cache_data
def run_customer_segmentation(_df):
    """KMeans customer segmentation (3 clusters)."""
    features = _df[["Customer_Age", "Sales", "Customer_Satisfaction"]]
    scaled = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled)
    return labels

@st.cache_data
def run_anomaly_detection(_df):
    """Isolation Forest anomaly detection on Sales."""
    iso = IsolationForest(contamination=0.03, random_state=42)
    return iso.fit_predict(_df[["Sales"]])

@st.cache_data
def run_sales_forecast(_df):
    """Holt-Winters 6-month sales forecast."""
    monthly = _df.groupby("Month")["Sales"].sum()
    monthly.index = pd.to_datetime(monthly.index)
    model = ExponentialSmoothing(monthly, trend="add").fit()
    forecast = model.forecast(6)
    return monthly, forecast

df["Customer_Segment"] = run_customer_segmentation(df).astype(str)
df["Anomaly"] = run_anomaly_detection(df)
monthly_sales, forecast = run_sales_forecast(df)


# ============================================================================
# STEP 4 – AGGREGATIONS FOR DASHBOARDS
# ============================================================================
daily_sales = df.groupby("Date")["Sales"].sum().reset_index()
product_sales = df.groupby("Product")["Sales"].sum().reset_index()
region_sales = df.groupby("Region")["Sales"].sum().reset_index()
anomaly_count = int((df["Anomaly"] == -1).sum())


# ============================================================================
# STEP 5 – RAG SYSTEM SETUP (Vector Store + Retriever)
# ============================================================================
@st.cache_resource
def build_vector_store(_documents):
    """Chunk knowledge base documents, embed, and store in FAISS."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(_documents)
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

vectorstore = build_vector_store(kb_documents)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


# ============================================================================
# CUSTOM RETRIEVER – pandas-based statistical extraction
# ============================================================================
def custom_data_retriever(query: str, _df: pd.DataFrame, _summary: dict) -> str:
    """
    A custom retriever that interprets a natural-language query and
    returns relevant statistics computed with pandas on-the-fly.
    Falls back to the FAISS retriever for general questions.
    """
    q = query.lower()

    # ── Pattern-based statistical retrieval ──
    if any(kw in q for kw in ["top product", "best product", "best selling"]):
        top = _summary["sales_by_product"].sort_values(ascending=False)
        return "Product Sales Ranking:\n" + top.to_string()

    if any(kw in q for kw in ["top region", "best region", "regional"]):
        top = _summary["sales_by_region"].sort_values(ascending=False)
        return "Regional Sales Ranking:\n" + top.to_string()

    if any(kw in q for kw in ["trend", "monthly", "over time"]):
        return "Monthly Sales Trend:\n" + _summary["sales_by_month"].to_string()

    if any(kw in q for kw in ["customer", "demographic", "age", "gender", "segment"]):
        result = "Sales by Age Group:\n" + _summary["sales_by_age_group"].to_string()
        result += "\n\nSales by Gender:\n" + _summary["sales_by_gender"].to_string()
        return result

    if any(kw in q for kw in ["statistic", "summary", "total", "average", "mean", "median"]):
        return (
            f"Total Sales: ${_summary['total_sales']:,.2f}\n"
            f"Average Sale: ${_summary['average_sales']:,.2f}\n"
            f"Median Sale: ${_summary['median_sales']:,.2f}\n"
            f"Std Dev: ${_summary['std_sales']:,.2f}"
        )

    if any(kw in q for kw in ["anomal", "outlier"]):
        n = int((_df["Anomaly"] == -1).sum())
        return f"Anomalies detected: {n} out of {len(_df)} records ({n/len(_df)*100:.1f}%)"

    if any(kw in q for kw in ["forecast", "predict", "future"]):
        return "6-Month Sales Forecast:\n" + forecast.to_string()

    # ── Fallback: FAISS semantic search ──
    docs = retriever.invoke(query)
    return "\n\n".join(d.page_content for d in docs)


# ============================================================================
# STEP 6 – LLM + CHAIN PROMPTS + MEMORY
# ============================================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# ── Chain 1: Query Classification (LCEL) ──
classify_prompt = PromptTemplate(
    input_variables=["user_query"],
    template=textwrap.dedent("""\
        Classify the following user query into one category:
        sales_trend, product_analysis, regional_analysis, customer_demographics, anomaly, forecast, general

        Query: {user_query}

        Respond with ONLY the category name, nothing else."""),
)
classify_chain = classify_prompt | llm | StrOutputParser()

# ── Chain 2: Insight Generation (LCEL) ──
insight_prompt = PromptTemplate(
    input_variables=["user_query", "category", "data_context"],
    template=textwrap.dedent("""\
        You are InsightForge, a senior AI business intelligence analyst.
        A user has asked a question classified as: {category}

        Relevant data context:
        {data_context}

        User question: {user_query}

        Provide a clear, data-driven executive response. Include specific numbers
        and actionable recommendations where appropriate. Format your response
        with bullet points for easy reading."""),
)
insight_chain = insight_prompt | llm | StrOutputParser()

# ── Conversation Memory (session_state, window of last 5 turns) ──
MEMORY_WINDOW = 5

if "messages" not in st.session_state:
    st.session_state.messages = []


def _get_chat_history_text() -> str:
    """Return the last MEMORY_WINDOW turns as a text block."""
    recent = st.session_state.messages[-(MEMORY_WINDOW * 2):]
    if not recent:
        return ""
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    )


def ask_insightforge(user_query: str) -> str:
    """
    Full RAG pipeline:
    1. Classify the query
    2. Retrieve relevant data (custom retriever + FAISS fallback)
    3. Generate insight with context + chat history
    """
    # Step 1 – Classify
    category = classify_chain.invoke({"user_query": user_query}).strip()

    # Step 2 – Retrieve data context
    data_context = custom_data_retriever(user_query, df, summary)

    # Step 3 – Add chat history for memory context
    history_text = _get_chat_history_text()
    full_context = data_context
    if history_text:
        full_context = f"Previous conversation:\n{history_text}\n\nCurrent data:\n{data_context}"

    # Step 4 – Generate insight
    insight = insight_chain.invoke({
        "user_query": user_query,
        "category": category,
        "data_context": full_context,
    })

    return insight


# ============================================================================
# STEP 7 – MODEL EVALUATION (LLM-based QA Grading)
# ============================================================================
_qa_eval_prompt = PromptTemplate(
    input_variables=["query", "answer", "result"],
    template=textwrap.dedent("""\
        You are a teacher grading a quiz.
        You are given a question, the student's answer, and the true answer.
        Grade the student answer as CORRECT or INCORRECT.
        Grade CORRECT if the student's answer contains the key facts from the true answer.
        Grade INCORRECT if the student's answer misses key facts or contradicts the true answer.

        Question: {query}
        True Answer: {answer}
        Student Answer: {result}

        Grade (respond with ONLY "CORRECT" or "INCORRECT"):"""),
)
_qa_eval_chain = _qa_eval_prompt | llm | StrOutputParser()


@st.cache_data
def run_evaluation():
    """Evaluate model responses against ground-truth QA pairs using LLM grading."""
    best_product = product_sales.sort_values("Sales", ascending=False).iloc[0]["Product"]
    best_region = region_sales.sort_values("Sales", ascending=False).iloc[0]["Region"]

    examples = [
        {"query": "Which product had the highest total sales?", "answer": best_product},
        {"query": "Which region performed best in total sales?", "answer": best_region},
        {"query": "How many total sales records are in the dataset?", "answer": str(len(df))},
        {"query": "What is the average customer satisfaction score?",
         "answer": f"{df['Customer_Satisfaction'].mean():.2f}"},
        {"query": "How many sales anomalies were detected?", "answer": str(anomaly_count)},
    ]

    # Generate predictions from our RAG system
    predictions = []
    for ex in examples:
        data_ctx = custom_data_retriever(ex["query"], df, summary)
        pred = insight_chain.invoke({
            "user_query": ex["query"],
            "category": "general",
            "data_context": data_ctx,
        })
        predictions.append({"query": ex["query"], "result": pred})

    # Grade each prediction with the LLM
    results = []
    for ex, pred in zip(examples, predictions):
        grade = _qa_eval_chain.invoke({
            "query": ex["query"],
            "answer": ex["answer"],
            "result": pred["result"],
        }).strip()
        results.append({
            "Question": ex["query"],
            "Expected Answer": ex["answer"],
            "Model Response": pred["result"][:200] + "..." if len(pred["result"]) > 200 else pred["result"],
            "Grade": grade,
        })
    return pd.DataFrame(results)


# ============================================================================
# STEP 8 – VISUALIZATION HELPERS
# ============================================================================
def chart_sales_trend():
    fig = px.line(
        daily_sales, x="Date", y="Sales",
        title="📈 Daily Sales Trend",
        template="plotly_dark",
    )
    fig.update_traces(line=dict(color="#38bdf8", width=1.5))
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
    )
    return fig


def chart_product_sales():
    fig = px.bar(
        product_sales.sort_values("Sales", ascending=True),
        x="Sales", y="Product", orientation="h",
        title="📦 Total Sales by Product",
        template="plotly_dark",
        color="Sales",
        color_continuous_scale="blues",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1", showlegend=False,
    )
    return fig


def chart_region_sales():
    fig = px.bar(
        region_sales.sort_values("Sales", ascending=True),
        x="Sales", y="Region", orientation="h",
        title="🌍 Total Sales by Region",
        template="plotly_dark",
        color="Sales",
        color_continuous_scale="teal",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1", showlegend=False,
    )
    return fig


def chart_segmentation():
    fig = px.scatter(
        df, x="Customer_Age", y="Sales",
        color="Customer_Segment",
        title="👥 Customer Segmentation (KMeans k=3)",
        template="plotly_dark",
        color_discrete_sequence=["#38bdf8", "#a78bfa", "#34d399"],
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
    )
    return fig


def chart_anomalies():
    df_plot = df.copy()
    df_plot["Status"] = df_plot["Anomaly"].map({1: "Normal", -1: "Anomaly"})
    fig = px.scatter(
        df_plot, x="Date", y="Sales", color="Status",
        title="🔍 Sales Anomaly Detection (Isolation Forest)",
        template="plotly_dark",
        color_discrete_map={"Normal": "#38bdf8", "Anomaly": "#f87171"},
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
    )
    return fig


def chart_forecast():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly_sales.index, y=monthly_sales.values,
        name="Actual", line=dict(color="#38bdf8"),
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast.values,
        name="Forecast", line=dict(color="#a78bfa", dash="dash"),
    ))
    fig.update_layout(
        title="🔮 6‑Month Sales Forecast (Holt‑Winters)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
    )
    return fig


def chart_gender_distribution():
    gender_data = df.groupby("Customer_Gender")["Sales"].sum().reset_index()
    fig = px.pie(
        gender_data, values="Sales", names="Customer_Gender",
        title="⚥ Sales by Gender",
        template="plotly_dark",
        color_discrete_sequence=["#38bdf8", "#a78bfa"],
        hole=0.4,
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
    )
    return fig


def chart_satisfaction_by_product():
    sat = df.groupby("Product")["Customer_Satisfaction"].mean().reset_index()
    fig = px.bar(
        sat.sort_values("Customer_Satisfaction", ascending=True),
        x="Customer_Satisfaction", y="Product", orientation="h",
        title="⭐ Avg Customer Satisfaction by Product",
        template="plotly_dark",
        color="Customer_Satisfaction",
        color_continuous_scale="greens",
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1", showlegend=False,
    )
    return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================
st.title("🔥 InsightForge – AI‑Powered Business Intelligence")
st.caption("Powered by GPT-4o-mini  •  RAG  •  LangChain  •  FAISS  •  Streamlit")

# ── Sidebar ──
with st.sidebar:
    st.header("📊 InsightForge")
    st.markdown("---")
    st.markdown(f"**Dataset:** {len(df):,} records")
    st.markdown(f"**Date range:** {df['Date'].min().strftime('%b %Y')} – {df['Date'].max().strftime('%b %Y')}")
    st.markdown(f"**Products:** {df['Product'].nunique()}")
    st.markdown(f"**Regions:** {df['Region'].nunique()}")
    st.markdown("---")
    st.markdown("### 🧠 RAG System")
    st.markdown(f"- Knowledge docs: {len(kb_documents)}")
    st.markdown(f"- Vector store: FAISS")
    st.markdown(f"- Memory window: last 5 turns")
    st.markdown("---")
    st.markdown(
        "Built for the **Purdue Generative AI Specialist** capstone.",
        unsafe_allow_html=True,
    )

# ── Tabs ──
tab_dashboard, tab_explorer, tab_assistant, tab_evaluation = st.tabs(
    ["📊 Dashboard", "🔎 Explorer", "🤖 AI Assistant", "📋 Evaluation"]
)

# ────────────────────────────────────────────────────────────────────────────
# TAB 1 – DASHBOARD
# ────────────────────────────────────────────────────────────────────────────
with tab_dashboard:
    st.subheader("Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Sales", f"${summary['total_sales']:,.0f}")
    kpi2.metric("Avg Sale", f"${summary['average_sales']:,.0f}")
    kpi3.metric("Avg Satisfaction", f"{df['Customer_Satisfaction'].mean():.2f}")
    kpi4.metric("Anomalies Detected", anomaly_count)

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(chart_sales_trend(), use_container_width=True)
    with col2:
        st.plotly_chart(chart_forecast(), use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(chart_product_sales(), use_container_width=True)
    with col4:
        st.plotly_chart(chart_region_sales(), use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        st.plotly_chart(chart_segmentation(), use_container_width=True)
    with col6:
        st.plotly_chart(chart_anomalies(), use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        st.plotly_chart(chart_gender_distribution(), use_container_width=True)
    with col8:
        st.plotly_chart(chart_satisfaction_by_product(), use_container_width=True)

# ────────────────────────────────────────────────────────────────────────────
# TAB 2 – EXPLORER (Interactive Filters)
# ────────────────────────────────────────────────────────────────────────────
with tab_explorer:
    st.subheader("🔎 Interactive Data Explorer")

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        selected_products = st.multiselect(
            "Filter by Product", df["Product"].unique(), default=df["Product"].unique()
        )
    with fcol2:
        selected_regions = st.multiselect(
            "Filter by Region", df["Region"].unique(), default=df["Region"].unique()
        )
    with fcol3:
        date_range = st.date_input(
            "Date Range",
            value=(df["Date"].min().date(), df["Date"].max().date()),
            min_value=df["Date"].min().date(),
            max_value=df["Date"].max().date(),
        )

    # Apply filters
    if len(date_range) == 2:
        mask = (
            df["Product"].isin(selected_products)
            & df["Region"].isin(selected_regions)
            & (df["Date"].dt.date >= date_range[0])
            & (df["Date"].dt.date <= date_range[1])
        )
    else:
        mask = df["Product"].isin(selected_products) & df["Region"].isin(selected_regions)

    filtered = df[mask]

    st.metric("Filtered Records", f"{len(filtered):,}")

    if len(filtered) > 0:
        ex1, ex2 = st.columns(2)
        with ex1:
            filt_daily = filtered.groupby("Date")["Sales"].sum().reset_index()
            fig = px.line(
                filt_daily, x="Date", y="Sales",
                title="Filtered Sales Trend", template="plotly_dark",
            )
            fig.update_traces(line=dict(color="#38bdf8"))
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1",
            )
            st.plotly_chart(fig, use_container_width=True)

        with ex2:
            filt_prod = filtered.groupby("Product")["Sales"].sum().reset_index()
            fig = px.bar(
                filt_prod.sort_values("Sales", ascending=True),
                x="Sales", y="Product", orientation="h",
                title="Filtered Product Sales", template="plotly_dark",
                color="Sales", color_continuous_scale="blues",
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                font_color="#cbd5e1", showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Filtered Data Sample")
        st.dataframe(filtered.head(100), use_container_width=True)
    else:
        st.warning("No data matches the selected filters.")

# ────────────────────────────────────────────────────────────────────────────
# TAB 3 – AI ASSISTANT (Chat with RAG + Memory)
# ────────────────────────────────────────────────────────────────────────────
with tab_assistant:
    st.subheader("🤖 Ask InsightForge")
    st.caption(
        "Chat with the AI assistant. It uses RAG (FAISS vector store), "
        "chain prompts, and conversation memory to answer your questions."
    )

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask about sales, products, regions, customers..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing data..."):
                response = ask_insightforge(user_input)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# ────────────────────────────────────────────────────────────────────────────
# TAB 4 – MODEL EVALUATION (QAEvalChain)
# ────────────────────────────────────────────────────────────────────────────
with tab_evaluation:
    st.subheader("📋 Model Evaluation – QAEvalChain")
    st.caption(
        "Evaluates the RAG system against ground-truth question-answer pairs "
        "using LangChain's QAEvalChain."
    )

    if st.button("🚀 Run Evaluation", type="primary"):
        with st.spinner("Running QAEvalChain evaluation (this may take a minute)..."):
            eval_df = run_evaluation()
        st.success("Evaluation complete!")
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
    else:
        st.info("Click **Run Evaluation** to assess the model against 5 ground‑truth QA pairs.")


# %% [markdown]
# ## Capstone Summary
#
# InsightForge integrates AI, ML, and LLMs to deliver enterprise‑grade
# business intelligence including forecasting, anomaly detection,
# customer segmentation, and natural‑language insights.
#
# **Technologies used:** LangChain, RAG (FAISS), GPT-4o-mini, Streamlit,
# Plotly, scikit-learn, statsmodels, pandas.
