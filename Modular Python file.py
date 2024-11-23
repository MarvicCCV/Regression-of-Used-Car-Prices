import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 100)

df_train= pd.read_csv('data/train.csv')


def basic_analysis(df):

    print("\n===== BASIC EXPLORATION OF DATA")
    print(f"\nShape of the data: {df.shape} ")
    print(f"\nThe training data has {df.shape[0]} rows and {df.shape[1]} columns")
    print("*"*20)
    print(df.info())
    print("**"*20)

    print("\nSummary statistics of categorical variables:")
    print(df.describe(include='object'))

    print("\nMissing values per column in the train dataset:")
    print(df.isna().sum())

    return df.describe()

basic_analysis(df_train)
