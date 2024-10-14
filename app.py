import streamlit as st
import pandas as pd
import numpy as np

# Set the title of the app
st.title("House Price Analysis")

# Load the dataset
# @st.cache  # Caches the data loading for performance
def load_data():
    data = pd.read_csv('melbourne.csv')
    return data

df = load_data()

# Remove all rows with any null values
df = df.dropna()

# Display dataset information
st.write("## Dataset Overview")
st.dataframe(df.head())

# Display summary statistics
st.write("## Summary Statistics")
st.write(df.describe())

# Sidebar for user inputs
st.sidebar.header("Filter Options")

# Example: Filter by number of bedrooms
bedrooms = st.sidebar.slider('Number of Bedrooms', int(df['Rooms'].min()), int(df['Rooms'].max()), (2, 4))
filtered_df = df[(df['Rooms'] >= bedrooms[0]) & (df['Rooms'] <= bedrooms[1])]

st.write(f"### Houses with {bedrooms[0]} to {bedrooms[1]} Bedrooms")
st.dataframe(filtered_df)

import matplotlib.pyplot as plt

st.write("## Distribution of House Prices")

fig, ax = plt.subplots()
ax.hist(df['Price'], bins=30, color='skyblue', edgecolor='black')
ax.set_xlabel('Price')
ax.set_ylabel('Frequency')
st.pyplot(fig)


import seaborn as sns
import matplotlib.pyplot as plt

st.write("## Price vs. Building Area")

# Ensure 'BuildingArea' exists and handle missing values
if 'BuildingArea' in filtered_df.columns:
    # Optionally, drop rows with missing values in 'BuildingArea' or 'Price'
    plot_df = filtered_df.dropna(subset=['BuildingArea', 'Price'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x='BuildingArea', y='Price', ax=ax, hue='Rooms', palette='viridis')
    ax.set_xlabel('Building Area (sq ft)')
    ax.set_ylabel('Price ($)')
    ax.set_title('Price vs. Building Area')
    st.pyplot(fig)
else:
    st.error("Column 'BuildingArea' does not exist in the dataset.")
