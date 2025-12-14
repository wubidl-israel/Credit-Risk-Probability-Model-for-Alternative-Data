import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np
import argparse


def preprocess(input_path, output_path):
    df=pd.read_csv(input_path)
    df['TransactionDate'] = pd.to_datetime(df['TransactionStartTime'])

    # Recency: Days since last transaction
    current_date = df['TransactionDate'].max()
 
    recency = df.groupby('CustomerId')['TransactionDate'].max().apply(lambda x: (current_date - x).days)

    # Frequency: Count of transactions
    frequency = df.groupby('CustomerId')['TransactionId'].count()

    # Monetary: Sum of transaction amounts
    monetary = df.groupby('CustomerId')['Amount'].sum()

    # Standard Deviation: Variability in transaction amounts
    std_dev = df.groupby('CustomerId')['Amount'].std().fillna(0)  # Replace NaN with 0

    # Combine RFMS metrics
    rfms = pd.DataFrame({
        'CustomerId': recency.index,
        'Recency': recency.values,
        'Frequency': frequency.values,
        'Monetary': monetary.values,
        'StandardDeviation': std_dev.values
    })

    # Standardize numerical columns
    scaler = StandardScaler()
    rfms[['Recency', 'Frequency', 'Monetary', 'StandardDeviation']] = scaler.fit_transform(
        rfms[['Recency', 'Frequency', 'Monetary', 'StandardDeviation']]
    )

    # Calculate RFMS_Score
    rfms['RFMS_Score'] = (
        0.25 * rfms['Recency'] + 
        0.25 * rfms['Frequency'] + 
        0.25 * rfms['Monetary'] + 
        0.25 * rfms['StandardDeviation']
    )

    # Assign Risk_Label based on median RFMS_Score
    threshold = rfms['RFMS_Score'].median()
    rfms['Risk_Label'] = rfms['RFMS_Score'].apply(lambda x: 'Good' if x >= threshold else 'Bad')

    # Create RFMS_Bin using pd.qcut
    rfms['RFMS_Bin'] = pd.qcut(rfms['RFMS_Score'], q=10, duplicates='drop')

    # Calculate WoE for each bin
    bins = rfms.groupby('RFMS_Bin').agg({
        'Risk_Label': lambda x: (x == 'Good').sum(),  # Count of 'Good' customers
        'CustomerId': 'count'  # Total customers in each bin
    }).rename(columns={'CustomerId': 'Total'})

    bins['Bad'] = bins['Total'] - bins['Risk_Label']  # Total - Good = Bad

    # Avoid division by zero in WoE
    smoothing_factor = 1e-5
    bins['Good_Percentage'] = (bins['Risk_Label'] + smoothing_factor) / bins['Risk_Label'].sum()
    bins['Bad_Percentage'] = (bins['Bad'] + smoothing_factor) / bins['Bad'].sum()

    bins['WoE'] = np.log(bins['Good_Percentage'] / bins['Bad_Percentage'])

    # Merge WoE back to rfms
    rfms = rfms.merge(bins[['WoE']], left_on='RFMS_Bin', right_index=True)
      # Encode categorical variables (e.g., 'RFMS_Bin')
    if 'RFMS_Bin' in rfms.columns:
        label_encoder = LabelEncoder()
        rfms['RFMS_Bin'] = label_encoder.fit_transform(rfms['RFMS_Bin'])
    
    # Encode target variable (y) if it's categorical
    if (rfms.dtypes == 'object').any():
        for col in rfms.select_dtypes(include=['object']).columns:
            rfms[col] = label_encoder.fit_transform(rfms[col])
    
    rfms.to_csv(output_path, index=False)



if __name__ == "__main__" :
    parser= argparse.ArgumentParser()    
    parser.add_argument("--input", required=True, help='Path to read csv')
    parser.add_argument('--output', required=True, help='Path to output csv')
    args=parser.parse_args()
    preprocess(args.input,args.output)

