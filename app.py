import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Mall Customer Clustering",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.stMetric {
    background-color: rgba(255, 255, 255, 0.1);
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #667eea;
}
h1 {
    color: #ffffff;
}
h2 {
    color: #ffffff;
}
.prediction-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
}
.cluster-info {
    background-color: rgba(255, 255, 255, 0.95);
    padding: 20px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# LOAD MODEL & SCALER
# ===============================
@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load("dt_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        return None, None

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    try:
        return pd.read_csv("clustered_mall_customers.csv")
    except FileNotFoundError:
        return None

model, scaler = load_model_and_scaler()
df = load_data()

# ===============================
# CLUSTER INFO
# ===============================
CLUSTER_INFO = {
    0: {
        "name": "High Value Customers",
        "description": "Young customers with high spending score and good income",
        "color": "#FF6B6B",
        "characteristics": ["Age: Young (25-40)", "Income: High (40-80k)", "Spending Score: High (70-100)"]
    },
    1: {
        "name": "Potential Target",
        "description": "Middle-aged customers with moderate to high spending",
        "color": "#4ECDC4",
        "characteristics": ["Age: Middle-aged (35-50)", "Income: Moderate (30-70k)", "Spending Score: Moderate to High (50-100)"]
    },
    2: {
        "name": "Average Customers",
        "description": "Young customers with low to moderate spending",
        "color": "#45B7D1",
        "characteristics": ["Age: Young (20-50)", "Income: Low (20-50k)", "Spending Score: Low to Moderate (20-60)"]
    },
    3: {
        "name": "Loyal Customers",
        "description": "Older customers with variable spending patterns",
        "color": "#FFA07A",
        "characteristics": ["Age: Older (40-70)", "Income: Variable", "Spending Score: Variable"]
    },
    4: {
        "name": "Budget Conscious",
        "description": "Customers with high income but low spending",
        "color": "#98D8C8",
        "characteristics": ["Age: Varied", "Income: High (50-150k)", "Spending Score: Low (10-50)"]
    }
}

# ===============================
# TITLE
# ===============================
st.markdown("""
<h1>🛍️ Mall Customer Clustering Prediction</h1>
<p style='color:white; text-align:center;'>
Predict customer segments and discover which group your customer belongs to!
</p>
""", unsafe_allow_html=True)

# ===============================
# STOP IF FILES MISSING
# ===============================
if model is None or scaler is None or df is None:
    st.error("Model or data files not found. Ensure they are in the same folder as app.py")
    st.stop()

# ===============================
# INPUT SECTION
# ===============================
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### 📊 Customer Information")
    age = st.slider("👤 Age", int(df['Age'].min()), int(df['Age'].max()), 30)
    annual_income = st.slider("💰 Annual Income (k$)",
                              int(df['Annual Income (k$)'].min()),
                              int(df['Annual Income (k$)'].max()), 50)
    spending_score = st.slider("🎯 Spending Score (1-100)", 1, 100, 50)

with col2:
    st.markdown("### 📈 Dataset Statistics")
    st.metric("Total Customers", len(df))
    st.metric("Average Age", f"{df['Age'].mean():.1f}")

# ===============================
# PREDICTION
# ===============================
if st.button("🚀 Predict Cluster", use_container_width=True):

    input_data = pd.DataFrame({
        "Age": [age],
        "Annual Income (k$)": [annual_income],
        "Spending Score (1-100)": [spending_score]
    })

    # ✅ USE TRAINED SCALER
    scaled_input = scaler.transform(input_data)
    cluster = model.predict(scaled_input)[0]
    cluster_details = CLUSTER_INFO[cluster]

    st.markdown(f"""
    <div class="prediction-box">
        <h2>Cluster Prediction</h2>
        <h1>{cluster}</h1>
        <h3>{cluster_details['name']}</h3>
    </div>
    """, unsafe_allow_html=True)

    # ===============================
    # CLUSTER INFO
    # ===============================
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 📋 Cluster Description")
        st.info(cluster_details["description"])

    with col_b:
        st.markdown("### 🎯 Key Characteristics")
        for c in cluster_details["characteristics"]:
            st.markdown(f"• {c}")

    # ===============================
    # VISUALIZATIONS
    # ===============================
    st.markdown("### 📊 Visualizations")

    fig = px.scatter_3d(
        df,
        x="Age",
        y="Annual Income (k$)",
        z="Spending Score (1-100)",
        color=df["Cluster"].astype(str),
        title="3D Cluster Distribution"
    )

    fig.add_scatter3d(
        x=[age], y=[annual_income], z=[spending_score],
        mode="markers",
        marker=dict(size=14, color="red"),
        name="Your Input"
    )

    st.plotly_chart(fig, use_container_width=True)

    cluster_counts = df["Cluster"].value_counts().sort_index()
    fig_pie = go.Figure(go.Pie(
        labels=[f"Cluster {i}" for i in cluster_counts.index],
        values=cluster_counts.values
    ))

    fig_pie.update_layout(title="Customer Distribution by Cluster")
    st.plotly_chart(fig_pie, use_container_width=True)

# ===============================
# FOOTER
# ===============================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:white;'>
🛍️ Mall Customer Clustering Analysis | Powered by K-Means & Decision Tree ML Models
</div>
""", unsafe_allow_html=True)
