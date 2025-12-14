import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np

logging.basicConfig(level=logging.INFO,format='%(asctime)s- %(levelname)s-%(message)s')

class EDA:
    #constructor
    def __init__(self, df):
        self.df =df
 
    #prints structure of the dataframe
    def structure(self):
        logging.info('Printing datasets shape....')
        return self.df.shape

    #Checks for any missing values
    def check_missing_values(self):
        logging.info('Counting and suming missing value....')
        return self.df.isnull().sum()

    # Prints datatypes
    def datatypes(self):
        logging.info('Printing data types....')
        return self.df.dtypes

    # Box plot of columns to see skewed values
    def skewed(self, col):
        logging.info('plotting skewed data....')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=self.df[col], color="skyblue")
        plt.title(f"Box Plot of {col}", fontsize=16)
        plt.xlabel(col, fontsize=14)
        plt.show()
   
    # statistical summary of columns
    def discription(self):
        logging.info('Calculating statistical summary....')
        return self.df.describe()
   
     # Numerical value distrbutions
    def numeric_dis(self, col):
        logging.info('Plotting numeric distribution....')
        plt.figure(figsize=(8, 5))
        plt.hist(self.df[col], bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        plt.title(f"Histogram of {col}", fontsize=16)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()

    # Catagorical value distrbutions
    def catagorical_dis(self, col):
        logging.info('Plotting catagorical distribution....')
        cat= self.df[col].value_counts()
        print(cat)
        plt.hist(cat, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
        plt.title(f"Histogram of {col}", fontsize=16)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
    
    # Prints Total debts and credit in the dataset
    def cred_dept(self):
        logging.info('Calculating total debt and credit....')
        
        # Calculating totals
        total_amount = self.df["Amount"].abs().sum()
        positive_total = self.df[self.df["Amount"] > 0]["Amount"].sum()  # Total debt
        negative_total = self.df[self.df["Amount"] < 0]["Amount"].sum()  # Total credit
        
        # Calculating percentages
        debt_percentage = (positive_total / total_amount) * 100
        credit_percentage = (abs(negative_total) / total_amount) * 100
        
        # Displaying results
        print(f"Total debt: {positive_total} ({debt_percentage:.2f}%)")
        print(f"Total credit: {abs(negative_total)} ({credit_percentage:.2f}%)")

    
    # Displays correlation
    def correlation(self):
        logging.info('Calculating correlation....')
        corr=self.df['Amount'].corr(self.df['Value'])
        print(corr)
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=self.df['Amount'], y=self.df['Value'], alpha=0.7)
        plt.title("Scatter Plot of Amount vs Value", fontsize=16)
        plt.xlabel("Amount", fontsize=14)
        plt.ylabel("Value", fontsize=14)
        plt.grid(True)
        plt.show()
