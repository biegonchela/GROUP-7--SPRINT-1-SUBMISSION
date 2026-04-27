import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Olympics_dataset.csv")
    
         #MILESTONE 1

# first 10 rows of the dataset
dataset.head(10)

# Shape (rows, columns)
print("Shape:", dataset.shape)

# Column names
print("Columns:", dataset.columns)

# Data types & non-null counts
print(dataset.info())

#descriptive statistics(mean, std, min, max)
print(dataset.describe())

#check for missing values
dataset.isnull().sum()

# Percentage missing
print((dataset.isnull().sum() / len(dataset)) * 100)

#Check for duplicates
print("Duplicate rows:", dataset.duplicated().sum())

# Percentage of missing values
total_rows = len(dataset)

missing_cols = ['Age', 'Height', 'Weight']

print("=== MISSING VALUES PERCENTAGE ===")
for col in missing_cols:
    missing = dataset[col].isnull().sum()
    percentage = (missing / total_rows) * 100
    print(f"{col}: {missing} missing values ({percentage:.2f}%)")

#removing duplicates
dataset = dataset.drop_duplicates()

  
           #MILESTONE 2
#  Medal — fill nulls with 'No Medal' 
dataset['Medal'] = dataset['Medal'].fillna('No Medal')

#  Age — fill with median since less affected by outliers
dataset['Age']=dataset['Age'].fillna(dataset['Age'].median())

#  Height — fill with median grouped by Sport 
dataset['Height'] = dataset.groupby('Sport')['Height'].transform(
    lambda x: x.fillna(x.median())
)

# 4. Weight 
dataset['Weight'] = dataset.groupby('Sport')['Weight'].transform(
    lambda x: x.fillna(x.median())
)

#check if all missing values are handled
dataset.isnull().sum()

#  fill any remaining nulls with the overall global median
dataset['Height'] = dataset['Height'].fillna(dataset['Height'].median())
dataset['Weight'] = dataset['Weight'].fillna(dataset['Weight'].median())

# Final check for missing values
dataset.isnull().sum()

print("DATASET OVERVIEW ")
print(f"Total Athletes Records: {len(dataset)}")
print(f"Unique Athletes: {dataset['Name'].nunique()}")
print(f"Unique Countries: {dataset['NOC'].nunique()}")
print(f"Unique Sports: {dataset['Sport'].nunique()}")
print(f"Years Covered: {dataset['Year'].min()} to {dataset['Year'].max()}")
print(f"Total Events: {dataset['Event'].nunique()}")
print("")
print("AGE STATISTICS ")
print(f"Average Age: {dataset['Age'].mean():.1f} years")
print(f"Youngest Athlete: {dataset['Age'].min():.0f} years")
print(f"Oldest Athlete: {dataset['Age'].max():.0f} years")

print("=== MEDAL DISTRIBUTION ===")
print(dataset['Medal'].value_counts())

medals_only = dataset[dataset['Medal'] != 'No Medal']
top_countries = medals_only.groupby('NOC')['Medal'].count().sort_values(ascending=False).head(10)
print(" TOP 10 COUNTRIES BY MEDALS ")
print(top_countries)

# Aggregation 
medals_per_country = dataset[dataset['Medal'] != 'No Medal'].groupby(
    ['NOC', 'Year'])['Medal'].count().reset_index()
medals_per_country.columns = ['NOC', 'Year', 'Total_Medals']
print(medals_per_country.head())

# Grouping 
avg_age_sport = dataset.groupby('Sport')['Age'].mean().reset_index()
avg_age_sport.columns = ['Sport', 'Average_Age']
avg_age_sport = avg_age_sport.sort_values('Average_Age', ascending=False)
print(avg_age_sport.head(10))

# Joining the aggregated data back into main dataset
dataset = dataset.merge(medals_per_country, on=['NOC', 'Year'], how='left')

      #FEATURE ENGINEERING
# Feature 1 — BMI 
dataset['BMI'] = dataset['Weight'] / ((dataset['Height'] / 100) ** 2)
dataset['BMI'] = dataset['BMI'].round(2)

# Feature 2 — Age group categories
bins = [0, 18, 25, 32, 40, 100]
labels = ['Junior', 'Young Adult', 'Prime', 'Experienced', 'Veteran']
dataset['Age_Group'] = pd.cut(dataset['Age'], bins=bins, labels=labels)

# Feature 3 — Medal winner flag (1 = won a medal, 0 = did not)
dataset['Medal_Winner'] = dataset['Medal'].apply(
    lambda x: 1 if x != 'No Medal' else 0)

# Feature 4 — Olympic Era
dataset['Era'] = pd.cut(dataset['Year'],
    bins=[1890, 1939, 1959, 1979, 1999, 2020],
    labels=['Early (pre-1940)', 'Post-War', 'Cold War', 'Modern', 'Contemporary'])

print(dataset[['BMI', 'Age_Group', 'Medal_Winner', 'Era']].head(10))

print("=== PIPELINE VERIFICATION ===")
print(f"Total Records: {len(dataset)}")
print(f"Total Columns (original + engineered): {dataset.shape[1]}")
print(f"\nNew columns added:")
print(['BMI', 'Age_Group', 'Medal_Winner', 'Era', 'Total_Medals'])
print(f"\nSample of engineered features:")
print(dataset[['Name', 'Sport', 'BMI', 'Age_Group', 'Medal_Winner', 'Era']].head())



