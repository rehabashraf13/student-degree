import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import os
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from groq import Groq
# =====================================================
# LOAD ENV
# =====================================================

from dotenv import load_dotenv
from groq import Groq
import os

load_dotenv(dotenv_path=".env")

API_KEY = os.getenv("GROQ_API_KEY")

print(API_KEY)

client = Groq(api_key=API_KEY)
# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Student Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
## =====================================================
# CUSTOM CSS
# =====================================================
st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.stApp {
    background: linear-gradient(to right, #0f172a, #020617);
    color: white;
}

section[data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #334155;
}

.block-container {
    padding-top: 2rem;
}

.metric-card {
    background: #111827;
    padding: 20px;
    border-radius: 18px;
}

div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827, #1e293b);
    border: 1px solid #38bdf8;
    padding: 20px;
    border-radius: 18px;
    box-shadow: 0px 0px 15px rgba(56,189,248,0.2);
}

[data-testid="stMetricValue"] {
    color: black !important;
    font-size: 42px !important;
    font-weight: 800 !important;
}

[data-testid="stMetricLabel"] {
    color: black !important;
    font-size: 18px !important;
    font-weight: 700 !important;
}
h1, h2, h3, h4 {
    color: white;
}

.stDataFrame {
    border-radius: 15px;
}

</style>
""",
    unsafe_allow_html=True
)
# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    df = pd.read_csv("students.csv")

    df.columns = df.columns.str.strip().str.lower()

    numeric_cols = [
        "percentage",
        "student_score",
        "total_score"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

    df = df.dropna()

    return df


df = load_data()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🎓 AI Student Analytics")
st.sidebar.markdown("---")

axis_filter = st.sidebar.multiselect(
    "Select Axis",
    options=df["axis"].unique(),
    default=df["axis"].unique()
)

student_filter = st.sidebar.multiselect(
    "Select Students",
    options=df["full_name"].unique(),
    default=df["full_name"].unique()
)

percentage_filter = st.sidebar.slider(
    "Percentage Range",
    0,
    100,
    (0, 100)
)

# =====================================================
# FILTER DATA
# =====================================================

filtered_df = df[
    (df["axis"].isin(axis_filter))
    & (df["full_name"].isin(student_filter))
    & (df["percentage"] >= percentage_filter[0])
    & (df["percentage"] <= percentage_filter[1])
]

# =====================================================
# AI FUNCTIONS
# =====================================================
def generate_ai_analysis(dataframe):

    if not API_KEY:
        return "⚠️ GROQ_API_KEY not found"

    summary = dataframe.describe(include="all").to_string()
    sample = dataframe.head(10).to_string()

    prompt = f"""
You are a professional educational data analyst.

Analyze this dataset professionally.

Dataset Summary:
{summary}

Sample Data:
{sample}

Generate:
1. Executive Summary
2. Key Insights
3. Weak Students
4. Recommendations
5. Future Predictions
6. Risk Analysis

Keep it professional and concise.
"""

    response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are an expert AI educational analyst."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    return response.choices[0].message.content


#########################
def ask_ai(question, dataframe):

    if not API_KEY:
        return "⚠️ GROQ_API_KEY not found"

    sample = dataframe.head(20).to_string()

    prompt = f"""
Dataset:
{sample}

Question:
{question}

Answer professionally.
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a professional AI data analyst."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.7
    )

    return response.choices[0].message.content
# =====================================================
# MACHINE LEARNING ANALYTICS
# =====================================================

def predict_future_scores(dataframe):

    temp_df = dataframe.copy()

    temp_df = temp_df.reset_index(drop=True)
    temp_df["index"] = temp_df.index

    X = temp_df[["index"]]
    y = temp_df["percentage"]

    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    temp_df["predicted_score"] = predictions

    score = r2_score(y, predictions)

    return temp_df, round(score, 3)


def student_clustering(dataframe):

    cluster_df = dataframe[
        ["percentage", "student_score"]
    ].copy()

    scaler = StandardScaler()

    scaled = scaler.fit_transform(cluster_df)

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    dataframe["cluster"] = kmeans.fit_predict(scaled)

    return dataframe

# =====================================================
# MENU
# =====================================================

selected = option_menu(
    menu_title=None,
    options=[
        "Dashboard",
        "AI Insights",
        "Predictions",
        "AI Chat"
    ],
    icons=[
        "bar-chart-fill",
        "cpu-fill",
        "graph-up-arrow",
        "chat-dots-fill"
    ],
    orientation="horizontal"
)

# =====================================================
# DASHBOARD
# =====================================================

if selected == "Dashboard":

    st.title("🎓 Professional AI Student Analytics Dashboard")

    total_students = filtered_df[
        "full_name"
    ].nunique()

    avg_percentage = round(
        filtered_df["percentage"].mean(),
        2
    )

    highest_score = round(
        filtered_df["percentage"].max(),
        2
    )

    lowest_score = round(
        filtered_df["percentage"].min(),
        2
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "👨‍🎓 Students",
        total_students
    )

    c2.metric(
        "📈 Average",
        avg_percentage
    )

    c3.metric(
        "🏆 Highest",
        highest_score
    )

    c4.metric(
        "📉 Lowest",
        lowest_score
    )

    style_metric_cards()

    st.markdown("---")

    # =================================================
    # CHARTS
    # =================================================

    col1, col2 = st.columns(2)

    axis_avg = filtered_df.groupby(
        "axis"
    )["percentage"].mean().reset_index()

    fig1 = px.bar(
        axis_avg,
        x="axis",
        y="percentage",
        color="axis",
        title="Average Performance by Axis",
        template="plotly_dark"
    )

    col1.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.histogram(
        filtered_df,
        x="percentage",
        nbins=20,
        title="Student Distribution",
        template="plotly_dark"
    )

    col2.plotly_chart(
        fig2,
        use_container_width=True
    )

    # =================================================

    col3, col4 = st.columns(2)

    top_students = filtered_df.sort_values(
        by="percentage",
        ascending=False
    ).head(10)

    fig3 = px.bar(
        top_students,
        x="full_name",
        y="percentage",
        color="percentage",
        title="Top 10 Students",
        template="plotly_dark"
    )

    col3.plotly_chart(
        fig3,
        use_container_width=True
    )

    axis_count = filtered_df[
        "axis"
    ].value_counts().reset_index()

    axis_count.columns = [
        "axis",
        "count"
    ]

    fig4 = px.pie(
        axis_count,
        names="axis",
        values="count",
        title="Axis Distribution",
        template="plotly_dark"
    )

    col4.plotly_chart(
        fig4,
        use_container_width=True
    )

    # =================================================

    st.markdown("---")

    st.subheader("📋 Full Dataset")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

# =====================================================
# AI INSIGHTS
# =====================================================

elif selected == "AI Insights":

    st.title("🧠 AI Insights & Recommendations")

    st.write(
        "Generate advanced AI-powered educational insights."
    )

    if st.button("🚀 Generate AI Report"):

        with st.spinner(
            "AI is analyzing your dataset..."
        ):

            try:

                result = generate_ai_analysis(
                    filtered_df
                )

                st.success(
                    "Analysis Generated Successfully"
                )

                st.markdown(result)

            except Exception as e:

                st.error(f"Error: {e}")

# =====================================================
# PREDICTIONS
# =====================================================

elif selected == "Predictions":

    st.title("📈 Predictive Analytics")

    predicted_df, score = predict_future_scores(
        filtered_df
    )

    st.metric(
        "🤖 Prediction Accuracy (R²)",
        score
    )

    fig5 = go.Figure()

    fig5.add_trace(
        go.Scatter(
            y=predicted_df["percentage"],
            mode='lines+markers',
            name='Actual Scores'
        )
    )

    fig5.add_trace(
        go.Scatter(
            y=predicted_df["predicted_score"],
            mode='lines+markers',
            name='Predicted Scores'
        )
    )

    fig5.update_layout(
        template='plotly_dark',
        title='Actual vs Predicted Scores'
    )

    st.plotly_chart(
        fig5,
        use_container_width=True
    )

    # =================================================
    # CLUSTERING
    # =================================================

    clustered = student_clustering(
        filtered_df.copy()
    )

    fig6 = px.scatter(
        clustered,
        x="student_score",
        y="percentage",
        color="cluster",
        hover_data=["full_name"],
        title="Student Performance Clustering",
        template="plotly_dark"
    )

    st.plotly_chart(
        fig6,
        use_container_width=True
    )

# =====================================================
# AI CHAT
# =====================================================

elif selected == "AI Chat":

    st.title("💬 AI Chat With Your Data")

    question = st.text_input(
        "Ask AI about your dataset"
    )

    if question:

        with st.spinner(
            "AI is thinking..."
        ):

            try:

                answer = ask_ai(
                    question,
                    filtered_df
                )

                st.markdown(answer)

            except Exception as e:

                st.error(f"Error: {e}")

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")

st.markdown(
    """
<center>
<h5>
🚀 Developed with AI Analytics • Streamlit • Gemini AI • Machine Learning
</h5>
</center>
""",
    unsafe_allow_html=True
)