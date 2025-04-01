import streamlit as st
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

# Set the title of the apps
st.title("House Price Analysis")

# Load the dataset
@st.cache_data  # Caches the data loading for performance
def load_data():
    data = pd.read_csv('melbourne.csv')
    return data

df = load_data()
# df = df.style.set_properties(**{'text-align': 'center'})

# df = df.set_table_styles([{
#     'selector': 'th',
#     'props': [('text-align', 'center')]
# }])


# 1. Display data types and dimensions
st.write("## Data Types and Dimensions")
st.write("### Shape of the dataset:")
st.write(f"Number of Rows: {df.shape[0]}")
st.write(f"Number of Columns: {df.shape[1]}")
st.write("### Data Types of Each Column:")
st.write(df.dtypes)


# 2. Data dictionary (explanation of columns)
st.write("## Data Dictionary")
data_dict = {
    "Suburb": "The name of the suburb where the house is located",
    "Address": "The address of the property",
    "Rooms": "Number of rooms in the house",
    "Type": "Type of dwelling (house, unit, etc.)",
    "Price": "Price of the property in AUD",
    "Method": "Method of sale (e.g., auction, private)",
    "SellerG": "Name of the real estate agency",
    "Date": "Date of sale",
    "Distance": "Distance from Melbourne CBD (in km)",
    "Postcode": "Postcode of the property",
    "Bedroom2": "Number of bedrooms (sometimes differs from 'Rooms')",
    "Bathroom": "Number of bathrooms",
    "Car": "Number of car parking spaces",
    "Landsize": "Land area of the property (in square meters)",
    "BuildingArea": "Building area of the property (in square meters)",
    "YearBuilt": "Year the property was built",
    "CouncilArea": "The local government area for the property",
    "Lattitude": "Latitude of the property",
    "Longtitude": "Longitude of the property",
    "Regionname": "The general region the property is located in",
    "Propertycount": "Number of properties in the suburb"
}
st.write(pd.DataFrame(list(data_dict.items()), columns=["Column", "Description"]))


# 3. Data distribution (scatter plots)
st.write("## Data Distribution: Scatter Plots")
# Create scatter plots for each numeric column against Price
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for column in numeric_columns:
    st.write(f"### {column} vs. Price")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=column, y='Price', ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Price')
    st.pyplot(fig)

# Create count plots for categorical columns
st.write("## Data Distribution: Count Plots for Categorical Columns")
categorical_columns = df.select_dtypes(include=['object']).columns

for column in categorical_columns:
    st.write(f"### Distribution of {column}")

    # Drop rows with missing values in the categorical column
    plot_df = df.dropna(subset=[column])
    
    # Check if the column has too many unique values and limit it for readability
    if plot_df[column].nunique() > 20:  # If more than 20 unique categories, display only the top 20
        top_20_categories = plot_df[column].value_counts().nlargest(20).index
        plot_df = plot_df[plot_df[column].isin(top_20_categories)]
        st.write(f"Displaying top 20 categories for {column}")
    
    fig, ax = plt.subplots()
    sns.countplot(data=plot_df, x=column, ax=ax)
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    plt.xticks(rotation=45)  # Rotate x-axis labels for better readability if needed
    st.pyplot(fig)


# 4. Percentage of missing values
st.write("## Percentage of Missing Values")
missing_percentage = (df.isnull().sum() / df.shape[0]) * 100
st.write(missing_percentage)


# 5. Data Cleaning (Removing null and duplicate values)
st.write("## Data Cleaning")
st.write("### Before Cleaning:")
st.write(f"Number of Missing Values: {df.isnull().sum().sum()}")
st.write(f"Number of Duplicates: {df.duplicated().sum()}")

# Remove null values
df_cleaned = df.dropna()
# Remove duplicate values
df_cleaned = df_cleaned.drop_duplicates()

st.write("### After Cleaning:")
st.write(f"Number of Missing Values: {df_cleaned.isnull().sum().sum()}")
st.write(f"Number of Duplicates: {df_cleaned.duplicated().sum()}")

# 6. Imputation (Just a note here)
st.write("## Imputation")
st.write("In this case, we've opted to remove rows with missing values. However, imputation techniques, such as filling missing values with the mean, median, or mode, could be considered based on the dataset's characteristics and the analysis goal.")

type_regionname_table = pd.crosstab(df['Type'],df['Regionname'])

from scipy.stats import chi2_contingency

chi2,p_value,dof,excepted = chi2_contingency(type_regionname_table)
# Display the results
st.write("## Relationship between House Type and Region")
st.write("### Contingency Table")
st.write(type_regionname_table)

st.write("### Chi-Square Test Results")
st.write(f"Chi-Square Statistic: {chi2}")
st.write(f"P-Value: {p_value}")



if p_value < 0.05:
    st.write("There is a significant relationship between house type and region (p < 0.05).")
else:
    st.write("There is no significant relationship between house type and region (p >= 0.05).")

# Filter by number of bedrooms
bedrooms = st.sidebar.slider('Number of Bedrooms', int(df['Rooms'].min()), int(df['Rooms'].max()), (2, 4))
filtered_df = df[(df['Rooms'] >= bedrooms[0]) & (df['Rooms'] <= bedrooms[1])]

st.write(f"### Houses with {bedrooms[0]} to {bedrooms[1]} Bedrooms")
st.dataframe(filtered_df)

# Distribution of house prices
st.write("## Distribution of House Prices")

fig, ax = plt.subplots()
ax.hist(df['Price'], bins=30, color='skyblue', edgecolor='black')
ax.set_xlabel('Price')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# Price vs. Building Area scatterplot
st.write("## Price vs. Building Area")

if 'BuildingArea' in filtered_df.columns:
    # Drop rows with missing values in 'BuildingArea' or 'Price'
    plot_df = filtered_df.dropna(subset=['BuildingArea', 'Price'])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x='BuildingArea', y='Price', ax=ax, hue='Rooms', palette='viridis')
    ax.set_xlabel('Building Area (sq ft)')
    ax.set_ylabel('Price ($)')
    ax.set_title('Price vs. Building Area')
    st.pyplot(fig)
else:
    st.error("Column 'BuildingArea' does not exist in the dataset.")

# Chi-Square Test: Relationship between house type and region
st.write("## Relationship between House Type and Region")

# Step 1: Create a contingency table for Type and Regionname
contingency_table = pd.crosstab(df['Type'], df['Regionname'])

# Step 2: Perform the chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

# Display the results
st.write("### Contingency Table")
st.write(contingency_table)

st.write("### Chi-Square Test Results")
st.write(f"Chi-Square Statistic: {chi2}")
st.write(f"P-Value: {p_value}")
st.write(f"Degrees of Freedom: {dof}")

# Interpret the result
if p_value < 0.05:
    st.write("There is a significant relationship between house type and region (p < 0.05).")
else:
    st.write("There is no significant relationship between house type and region (p >= 0.05).")
