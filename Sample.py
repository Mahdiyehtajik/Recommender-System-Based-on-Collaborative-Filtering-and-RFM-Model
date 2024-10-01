import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Create sample data for a clothing store
num_customers = 1000
current_date = datetime.now()

data = {
    'CustomerID': range(1, num_customers + 1),
    'LastPurchaseDate': [current_date - timedelta(days=np.random.randint(1, 365)) for _ in range(num_customers)],
    'PurchaseFrequency': np.random.randint(1, 20, num_customers),
    'TotalSpent': np.random.randint(50, 5000, num_customers)
}

df = pd.DataFrame(data)

# Calculate Recency
df['Recency'] = (current_date - df['LastPurchaseDate']).dt.days

# Function to create ranks with error handling
def create_ranks(series, num_bins=5, labels=None):
    try:
        return pd.qcut(series, q=num_bins, labels=labels, duplicates='drop')
    except ValueError:
        return pd.cut(series.rank(method='first'), bins=num_bins, labels=labels, include_lowest=True)

# Normalizing RFM scores
df['R_rank'] = create_ranks(df['Recency'], labels=[5, 4, 3, 2, 1])
df['F_rank'] = create_ranks(df['PurchaseFrequency'], labels=[1, 2, 3, 4, 5])
df['M_rank'] = create_ranks(df['TotalSpent'], labels=[1, 2, 3, 4, 5])

# Calculate final RFM score
df['RFM_Score'] = df[['R_rank', 'F_rank', 'M_rank']].sum(axis=1)

# Create sample product categories and their base recommendation scores
product_categories = ['T-shirts', 'Jeans', 'Dresses', 'Shoes', 'Accessories']
base_scores = {category: np.random.rand(num_customers) for category in product_categories}

# Adjust recommendations based on RFM score
def adjust_recommendations(rfm_scores, base_scores):
    max_rfm = rfm_scores.max()
    return {category: scores * (rfm_scores / max_rfm) for category, scores in base_scores.items()}

adjusted_recommendations = adjust_recommendations(df['RFM_Score'].values, base_scores)

# Get top 3 recommended categories for each customer
def get_top_recommendations(adjusted_recommendations, n=3):
    all_scores = pd.DataFrame(adjusted_recommendations)
    return all_scores.apply(lambda row: row.nlargest(n).index.tolist(), axis=1)

df['TopRecommendations'] = get_top_recommendations(adjusted_recommendations)

# Create customer segments
def create_customer_segments(rfm_score):
    if rfm_score >= 13:
        return 'Best Customers'
    elif 9 <= rfm_score < 13:
        return 'Loyal Customers'
    elif 5 <= rfm_score < 9:
        return 'Potential Churners'
    else:
        return 'Lost Customers'

df['CustomerSegment'] = df['RFM_Score'].apply(create_customer_segments)

# Print results
print("Sample Customer Data:")
print(df[['CustomerID', 'Recency', 'PurchaseFrequency', 'TotalSpent', 'RFM_Score', 'TopRecommendations']].head(10))

segment_counts = df['CustomerSegment'].value_counts()
print("\nCustomer Segments:")
print(segment_counts)

segment_analysis = df.groupby('CustomerSegment').agg({
    'Recency': 'mean',
    'PurchaseFrequency': 'mean',
    'TotalSpent': 'mean'
}).round(2)

print("\nSegment Analysis:")
print(segment_analysis)