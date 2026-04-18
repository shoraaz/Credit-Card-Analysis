

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Advanced Customer Segmentation",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA LOADING & PREPROCESSING ---
@st.cache_data
def load_data() -> pd.DataFrame:
    try:
        return pd.read_csv("credit_scoring.csv")
    except FileNotFoundError:
        st.error("Error: 'credit_scoring.csv' not found. Please ensure the dataset is in the same directory.")
        st.stop()

@st.cache_data
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    
    # Define mappings for categorical features
    education_level_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    employment_status_mapping = {'Unemployed': 0, 'Employed': 1, 'Self-Employed': 2}
    
    # Apply mappings
    if 'Education Level' in data.columns:
        data['Education Level'] = data['Education Level'].map(education_level_mapping)
    if 'Employment Status' in data.columns:
        data['Employment Status'] = data['Employment Status'].map(employment_status_mapping)
        
    # Calculate a custom feature (Credit Score approximation)
    data['Credit Score'] = (
        data.get('Payment History', 0) * 0.35 + 
        data.get('Credit Utilization Ratio', 0) * 0.30 + 
        data.get('Number of Credit Accounts', 0) * 0.15 + 
        data.get('Education Level', 0) * 0.10 + 
        data.get('Employment Status', 0) * 0.10
    )
    return data

@st.cache_data
def perform_clustering(data: pd.DataFrame, n_clusters: int):
    # Select features for clustering
    features = [
        'Credit Utilization Ratio', 
        'Payment History', 
        'Number of Credit Accounts', 
        'Loan Amount', 
        'Interest Rate'
    ]
    
    # Filter only available features
    available_features = [f for f in features if f in data.columns]
    X = data[available_features]
    
    # 1. Scaling (Crucial for distance-based algorithms like KMeans)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. KMeans Clustering
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 3. PCA for Visualization (Reduce to 3 components)
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(X_scaled)
    
    # Add results back to dataframe
    data['Cluster'] = clusters.astype(str)
    data['PCA1'] = pca_result[:, 0]
    data['PCA2'] = pca_result[:, 1]
    data['PCA3'] = pca_result[:, 2]
    
    return data, available_features, kmeans.cluster_centers_, scaler

# --- MAIN APP LAYOUT ---
def main():
    # Sidebar
    st.sidebar.header("⚙️ Dashboard Controls")
    n_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=6, value=4)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Pro Tip:** Advanced segmentation uses multiple features (Utilization, History, Loan Amount, etc.) scaled via `StandardScaler` and reduced via `PCA`."
    )

    # Load and process data
    raw_data = load_data()
    data = preprocess_data(raw_data)
    
    # Top Level Metrics
    st.title("📊 Advanced Customer Credit Segmentation")
    st.markdown("Identify high-value customers and mitigate risk using Machine Learning (KMeans & PCA).")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(data):,}")
    col2.metric("Avg Credit Score", f"{data['Credit Score'].mean():.2f}")
    col3.metric("Avg Loan Amount", f"${data['Loan Amount'].mean():,.2f}")
    col4.metric("Features Analyzed", f"{len(data.columns)}")

    # Tabs for organization
    tab1, tab2, tab3 = st.tabs(["🔍 Exploratory Data Analysis", "🤖 ML Segmentation (3D)", "🎯 Cluster Profiling"])

    # TAB 1: EDA
    with tab1:
        st.subheader("Data Overview")
        st.dataframe(data.head(10), use_container_width=True)
        
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(data, x='Loan Amount', nbins=30, title='Distribution of Loan Amounts', color_discrete_sequence=['#3366CC'])
            st.plotly_chart(fig_hist, use_container_width=True)
        with c2:
            fig_box = px.box(data, y='Credit Utilization Ratio', title='Credit Utilization Spread', color_discrete_sequence=['#DC3912'])
            st.plotly_chart(fig_box, use_container_width=True)
            
        st.subheader("Feature Correlations")
        numeric_df = data.select_dtypes(include=[np.number])
        fig_corr = px.imshow(numeric_df.corr(), color_continuous_scale='RdBu_r', aspect='auto')
        st.plotly_chart(fig_corr, use_container_width=True)

    # TAB 2: Clustering
    with tab2:
        segmented_data, features_used, centers, scaler = perform_clustering(data, n_clusters)
        
        st.subheader(f"3D PCA Projection ({n_clusters} Clusters)")
        st.markdown("Visualizing multi-dimensional customer segments mapped to 3 components.")
        
        fig_3d = px.scatter_3d(
            segmented_data, x='PCA1', y='PCA2', z='PCA3', 
            color='Cluster',
            hover_data=['Credit Score', 'Loan Amount'],
            opacity=0.7,
            title=f"Customer Segments (PCA Reduced)"
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_3d, use_container_width=True)

    # TAB 3: Insights
    with tab3:
        st.subheader("Cluster Behavioral Profiles")
        
        # Calculate mean for each feature per cluster
        cluster_means = segmented_data.groupby('Cluster')[features_used].mean().reset_index()
        
        # Radar Chart
        categories = features_used
        fig_radar = go.Figure()
        
        for i, row in cluster_means.iterrows():
            # Standardize for Radar chart visibility (0 to 1 scale)
            values = scaler.transform([row[features_used].values])[0] 
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=f"Cluster {row['Cluster']}"
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[-3, 3])),
            showlegend=True,
            title="Standardized Feature Means per Cluster"
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Business Action Items
        st.subheader("📑 Recommended Actions")
        st.info("**What's Next?** Look at the Radar chart. Clusters spiking in 'Credit Utilization' and dropping in 'Payment History' represent High Risk. Conversely, those with strong payment histories are Prime candidates for cross-selling premium loan products.")
        
        st.dataframe(cluster_means, use_container_width=True)

if __name__ == "__main__":
    main()

