import pandas as pd
import numpy as np

# Create sample data
np.random.seed(0)
data = {
    'CustomerID': range(1, 101),
    'Recency': np.random.randint(1, 100, 100),
    'Frequency': np.random.randint(1, 10, 100),  # Smaller range to increase chances of duplicates
    'Monetary': np.random.randint(100, 5000, 100)
}
rfm = pd.DataFrame(data)

# Function to create ranks with error handling
def create_ranks(series, num_bins=3, labels=None):
    try:
        return pd.qcut(series, q=num_bins, labels=labels, duplicates='drop')
    except ValueError:
        # If qcut fails, use rank instead
        return pd.cut(series.rank(method='first'), bins=num_bins, labels=labels, include_lowest=True)

# Normalizing RFM scores
rfm['R_rank'] = create_ranks(rfm['Recency'], labels=[3, 2, 1])
rfm['F_rank'] = create_ranks(rfm['Frequency'], labels=[1, 2, 3])
rfm['M_rank'] = create_ranks(rfm['Monetary'], labels=[1, 2, 3])

# Calculate final RFM score
rfm['RFM_Score'] = rfm[['R_rank', 'F_rank', 'M_rank']].sum(axis=1)

# Create sample predictions (this would normally come from your recommendation system)
predictions = np.random.rand(100)

# Adjust recommendations based on RFM score
def adjust_recommendations(rfm_scores, predictions):
    return predictions * (rfm_scores / rfm_scores.max())

adjusted_recommendations = adjust_recommendations(rfm['RFM_Score'].values, predictions)

# Print the first few rows of the result
print(rfm.head())
print("\nAdjusted Recommendations (first 5):")
print(adjusted_recommendations[:5])