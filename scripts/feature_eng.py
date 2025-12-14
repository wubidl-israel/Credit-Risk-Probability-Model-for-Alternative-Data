import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df= pd.read_csv('./data/data.csv')
def per_transaction(df):
    df['Debit'] = df['Amount'].apply(lambda x: x if x > 0 else 0)  # Positive amounts
    df['Credit'] = df['Amount'].apply(lambda x: abs(x) if x < 0 else 0)
    df= df.groupby('CustomerId').agg(
        TotalTransactionAmount=('Amount', 'sum'),
        AverageTransactionAmount=('Amount', 'mean'),
        TransactionCount=('TransactionId', 'count'),
        StdDevTransactionAmount=('Amount', 'std'),
        TotalDebit=('Debit', 'sum'),
        TotalCredit=('Credit', 'sum'),
        FirstTransaction=('TransactionStartTime', 'min'),  # Earliest transaction date
        LastTransaction=('TransactionStartTime', 'max'),
        PricingStrategyMode=('PricingStrategy', lambda x: x.mode()[0] if not x.mode().empty else None),
        FraudCount=('FraudResult', 'sum'),
        prodCat= ('ProductCategory',lambda x: x.mode()[0] if not x.mode().empty else None)
 
    ).reset_index()
    df['FirstTransaction']= pd.to_datetime(df['FirstTransaction'])
    df['start_year']= df['FirstTransaction'].dt.year
    df['start_month']= df['FirstTransaction'].dt.month
    df['start_day'] =df['FirstTransaction'].dt.day
    df['start_hour'] =df['FirstTransaction'].dt.hour

    df['LastTransaction']= pd.to_datetime(df['LastTransaction'])
    df['last_year']= df['LastTransaction'].dt.year
    df['last_month']= df['LastTransaction'].dt.month
    df['last_day'] =df['LastTransaction'].dt.day
    df['last_hour'] =df['LastTransaction'].dt.hour
    
    columns_to_encode = ['prodCat', 'PricingStrategyMode']



    df = pd.get_dummies(df, columns=columns_to_encode)
    #scaler= StandardScaler()
    #drops=['FirstTransaction','LastTransaction','CustomerId']
    #df_e=df_e.drop(columns=drops)
    #df_new= scaler.fit_transform(df_e)
    return df
df_edited=per_transaction(df)

def tryy(df):
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
    
    print(rfms)
tryy(df)