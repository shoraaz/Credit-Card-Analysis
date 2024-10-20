

# Import necessary libraries
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans

# Set up the Streamlit app
st.title("Customer Credit Score Segmentation")


# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("credit_scoring.csv")


data = load_data()

# Display the first few rows of the dataset
st.subheader("Dataset Preview")
st.write(data.head())

# Display dataset information
st.subheader("Dataset Information")
st.write(data.info())

# Display descriptive statistics
st.subheader("Descriptive Statistics")
st.write(data.describe())

# Visualize the distribution of the Credit Utilization Ratio
st.subheader("Credit Utilization Ratio Distribution")
credit_utilization_fig = px.box(data, y='Credit Utilization Ratio', title='Credit Utilization Ratio Distribution')
st.plotly_chart(credit_utilization_fig)

# Visualize the distribution of the Loan Amount
st.subheader("Loan Amount Distribution")
loan_amount_fig = px.histogram(data, x='Loan Amount', nbins=20, title='Loan Amount Distribution')
st.plotly_chart(loan_amount_fig)

# Create a correlation heatmap for numeric variables
st.subheader("Correlation Heatmap")
numeric_df = data[
    ['Credit Utilization Ratio', 'Payment History', 'Number of Credit Accounts', 'Loan Amount', 'Interest Rate',
     'Loan Term']]
correlation_fig = px.imshow(numeric_df.corr(), title='Correlation Heatmap')
st.plotly_chart(correlation_fig)

# Define mappings for categorical features
education_level_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
employment_status_mapping = {'Unemployed': 0, 'Employed': 1, 'Self-Employed': 2}

# Apply mappings to the dataset
data['Education Level'] = data['Education Level'].map(education_level_mapping)
data['Employment Status'] = data['Employment Status'].map(employment_status_mapping)

# Calculate credit scores using a FICO-like formula
credit_scores = []
for index, row in data.iterrows():
    payment_history = row['Payment History']
    credit_utilization_ratio = row['Credit Utilization Ratio']
    number_of_credit_accounts = row['Number of Credit Accounts']
    education_level = row['Education Level']
    employment_status = row['Employment Status']

    # Apply the weighted FICO formula
    credit_score = (payment_history * 0.35) + (credit_utilization_ratio * 0.30) + (number_of_credit_accounts * 0.15) + (
                education_level * 0.10) + (employment_status * 0.10)
    credit_scores.append(credit_score)

# Add the calculated credit scores to the dataset
data['Credit Score'] = credit_scores

# Display the first few rows with credit scores
st.subheader("Credit Scores")
st.write(data.head())

# Segment customers using KMeans clustering based on their credit scores
st.subheader("Customer Segmentation")
num_clusters = st.slider("Number of clusters", min_value=2, max_value=4, value=4)
X = data[['Credit Score']]
kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=42)
kmeans.fit(X)
data['Segment'] = kmeans.labels_

# Convert the 'Segment' column to a categorical data type
data['Segment'] = data['Segment'].astype('category')

# Visualize the segmentation based on credit scores
segmentation_fig = px.scatter(data, x=data.index, y='Credit Score', color='Segment',
                              title='Customer Segmentation based on Credit Scores',
                              color_discrete_sequence=['green', 'blue', 'yellow', 'red'])
st.plotly_chart(segmentation_fig)

# Map the segment labels to descriptive names
st.subheader("Segment Naming")
segment_names = {2: 'Very Low', 0: 'Low', 1: 'Good', 3: 'Excellent'}
data['Segment'] = data['Segment'].map(segment_names)

# Display the final segmented dataset
st.subheader("Segmented Dataset")
st.write(data)
