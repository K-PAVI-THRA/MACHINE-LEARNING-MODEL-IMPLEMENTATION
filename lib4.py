# Importing necessary libraries
import pandas as pd

# Load the dataset
df = pd.read_csv('spam.csv', encoding='latin-1')

# Displaying the first few rows of the dataset
df = df[['v1', 'v2']]  # Selecting relevant columns (label and message)
df.columns = ['label', 'message']  # Renaming columns for easier access
df.head()  # Displaying first 5 rows
