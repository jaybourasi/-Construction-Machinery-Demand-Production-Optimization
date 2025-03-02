import pandas as pd
import pprint

# ======================= Project - Demand & Production Optimization =======================

# Load the dataset
file_path = r"C:\Users\HP\Desktop\Demand & Production Optimization\Dataset Raw.csv"
df = pd.read_csv(file_path)

# ======================= Data Overview =======================

# Store Data Types
data_types = df.dtypes

# Convert numerical columns to integer without filling NaN values
num_cols = ['Order_ID', 'Order_Quantity', 'Inventory_Level', 'Lead_Time_Days', 'Delay_Days']
df[num_cols] = df[num_cols].astype('Int64')  # Uses pandas nullable integer type

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")

data_types = df.dtypes

# Check Data Overview
df.info()  # Column data types and missing values
Top_5_rows = df.head()  # First few rows
summary_numeric_columns = df.describe()  # Summary statistics for numerical columns

# Store Basic Stats in Variables
num_rows, num_cols = df.shape  # Number of rows and columns
column_names = df.columns.tolist()  # List of column names
missing_values = df.isnull().sum()  # Count of missing values per column

# ======================= Missing Values Count =======================

Total_missing_values_count = df.isnull().sum().sum()  # Total missing values in dataset
missing_percentage = (df.isnull().sum() / len(df)) * 100  # Missing data percentage

# ======================= Check for Duplicates =======================

duplicates_count = df.duplicated().sum()  # Count duplicate rows
df = df.drop_duplicates()  # Remove duplicate rows if needed

# ======================= Numeric Columns Analysis =======================

numerical_columns = ['Order_Quantity', 'Inventory_Level', 'Lead_Time_Days', 'Delay_Days']
numerical_summary = df[numerical_columns].describe()  # Summary statistics

data_types = df.dtypes

# ======================= Column-wise Summary for Numeric Data =======================

column_summary = []

for col in numerical_columns:
    # Calculate statistics
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()

    # Append details to the summary
    column_summary.append({
        "Column": col,
        "Data Type": df[col].dtype,
        "Null Count": df[col].isnull().sum(),
        "Not Null Count": df[col].notnull().sum(),
        "Missing Percentage": round(df[col].isnull().sum() / len(df) * 100, 2),
        "Mean": df[col].mean(skipna=True),
        "Median": df[col].median(skipna=True),
        "Std Dev": df[col].std(skipna=True),
        "Min": df[col].min(skipna=True),
        "Max": df[col].max(skipna=True),
        "Skewness": df[col].skew(skipna=True),
        "Kurtosis": df[col].kurtosis(skipna=True),
        "Outliers Count": outliers
    })

# Convert to DataFrame
column_summary_df = pd.DataFrame(column_summary)

# Export to CSV
column_summary_df.to_csv("eda_column_summary_with_outliers.csv", index=False)

print("EDA column summary with outliers, skewness, and kurtosis exported successfully! ✅")


################# CATEGORICAL COLUMN ANALYSIS ##################

# Identify categorical columns
categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

# Summary storage
categorical_summary = []

# Define high-cardinality threshold
high_cardinality_threshold = 50  

# Total count of rows in the dataset
total_rows = len(df)

for col in categorical_columns:
    unique_values = df[col].nunique()
    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else None
    missing_count = df[col].isnull().sum()
    not_null_count = df[col].notnull().sum()
    missing_percentage = round((missing_count / total_rows) * 100, 2)
    is_high_cardinality = "Yes" if unique_values > high_cardinality_threshold else "No"

    # Get top 5 frequent values with counts & percentage
    top_values = df[col].value_counts().head(5)
    top_values_dict = top_values.to_dict()
    top_values_percentage = (top_values / total_rows * 100).round(2).to_dict()

    # Store summary
    categorical_summary.append({
        "Column": col,
        "Total Count": total_rows,
        "Unique Values": unique_values,
        "Mode": mode_value,
        "Top 5 Frequent Values": top_values_dict,
        "Top 5 Percentage Distribution": top_values_percentage,
        "Null Count": missing_count,
        "Not Null Count": not_null_count,
        "Missing Percentage": missing_percentage,
        "High Cardinality": is_high_cardinality
    })

# Convert to DataFrame
categorical_summary_df = pd.DataFrame(categorical_summary)

# Export to CSV
categorical_summary_df.to_csv("eda_categorical_summary.csv", index=False)

print("EDA categorical column summary exported successfully! ✅")

# Cheking for Variables Dependency
from scipy.stats import chi2_contingency

cross_tab = pd.crosstab(df['Production_Status'], df['Change_Type'])
chi2, p, dof, expected = chi2_contingency(cross_tab)

if p < 0.05:
    print("✅ Variables are dependent!")
else:
    print("❌ Variables are independent!")



# ============ Distribution of Numerical Columns =============
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define numerical columns
numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

# Set plot style
sns.set_style("whitegrid")

# Iterate over each numerical column
for col in numerical_columns:
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # Calculate statistics
    mean_val = df[col].mean()
    median_val = df[col].median()
    skewness = df[col].skew()
    kurtosis = df[col].kurtosis()
    
    ## 1️⃣ Histogram with Mean & Median Labels
    sns.histplot(df[col], bins=30, kde=True, ax=axes[0, 0], color="blue")
    axes[0, 0].axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[0, 0].axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    axes[0, 0].set_title(f"Histogram of {col}", fontsize=14)
    axes[0, 0].set_xlabel(col, fontsize=12)
    axes[0, 0].set_ylabel("Frequency", fontsize=12)
    axes[0, 0].legend()

    ## 2️⃣ Boxplot with Labels
    sns.boxplot(y=df[col], ax=axes[0, 1], color="orange")
    axes[0, 1].set_title(f"Boxplot of {col}", fontsize=14)
    axes[0, 1].set_ylabel(col, fontsize=12)

    # Annotate median value
    axes[0, 1].text(0, median_val, f'Median: {median_val:.2f}', 
                    ha='left', fontsize=10, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7))

    ## 3️⃣ Density Plot (KDE)
    sns.kdeplot(df[col], fill=True, ax=axes[0, 2], color="green")
    axes[0, 2].set_title(f"Density Plot of {col}", fontsize=14)
    axes[0, 2].set_xlabel(col, fontsize=12)
    axes[0, 2].set_ylabel("Density", fontsize=12)

    ## 4️⃣ Violin Plot with Labels
    sns.violinplot(y=df[col], ax=axes[1, 0], color="purple")
    axes[1, 0].set_title(f"Violin Plot of {col}", fontsize=14)
    axes[1, 0].set_ylabel(col, fontsize=12)

    # Annotate min, max values
    axes[1, 0].text(0, df[col].min(), f'Min: {df[col].min():.2f}', 
                    ha='right', fontsize=10, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7))
    axes[1, 0].text(0, df[col].max(), f'Max: {df[col].max():.2f}', 
                    ha='right', fontsize=10, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7))

    ## 5️⃣ Log Transformation for Skewed Data
    if abs(skewness) > 1:  # Apply log transformation for highly skewed data
        log_col = np.log1p(df[col])  # log1p to avoid log(0) issue
        sns.histplot(log_col, bins=30, kde=True, ax=axes[1, 1], color="brown")
        axes[1, 1].set_title(f"Log-Transformed Histogram of {col}", fontsize=14)
        axes[1, 1].set_xlabel(f"Log of {col}", fontsize=12)
        axes[1, 1].set_ylabel("Frequency", fontsize=12)

    ## 6️⃣ Skewness & Kurtosis Annotations
    axes[1, 2].text(0.1, 0.5, f'Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}',
                    fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    axes[1, 2].set_title(f"Skewness & Kurtosis for {col}", fontsize=14)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


# ================ Pre-processing Steps ====================
# Check missing values in numeric columns
missing_values_numeric = df.select_dtypes(include=['number']).isnull().sum()

# ================= Handling Missing Data ====================
df = df.dropna(subset=['Order_ID'])
df = df.dropna(subset=['Date'])

from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
import numpy as np

# Log transform for skewed columns
skewed_cols = ['Order_Quantity', 'Inventory_Level', 'Lead_Time_Days', 'Delay_Days']
df[skewed_cols] = df[skewed_cols].apply(lambda x: np.log1p(x) if x.isnull().sum() > 0 else x)

# Select numerical columns for imputation
num_cols = ['Order_Quantity', 'Inventory_Level', 'Lead_Time_Days', 'Delay_Days']
features = df[num_cols].copy()

# KNN Imputation for missing values
knn_imputer = KNNImputer(n_neighbors=5)
df[num_cols] = knn_imputer.fit_transform(features)

# Reverse log transformation
df[skewed_cols] = df[skewed_cols].apply(lambda x: np.expm1(x))


# ================ For Categorical Data =======================
# Select categorical columns
categorical_columns = df.select_dtypes(include=['object']).columns

# Find missing values in categorical columns
missing_categorical = df[categorical_columns].isnull().sum()



categorical_columns = df.select_dtypes(include=['object']).columns

# Apply forward fill (ffill) for each categorical column
df[categorical_columns] = df[categorical_columns].fillna(method='ffill')
# Apply backward fill (ffill) for each categorical column
df[categorical_columns] = df[categorical_columns].fillna(method='bfill')


#========================= Outliers =====================
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

# Define lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter outliers
df_no_outliers = df[~((df[numerical_columns] < lower_bound) | (df[numerical_columns] > upper_bound)).any(axis=1)]


# ============ Distribution of Numerical Columns (After Removing Outliers) =============
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define numerical columns
numerical_columns = df_no_outliers.select_dtypes(include=['number']).columns.tolist()

# Set plot style
sns.set_style("whitegrid")

# Iterate over each numerical column
for col in numerical_columns:
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    # Calculate statistics
    mean_val = df_no_outliers[col].mean()
    median_val = df_no_outliers[col].median()
    skewness = df_no_outliers[col].skew()
    kurtosis = df_no_outliers[col].kurtosis()
    
    ## 1️⃣ Histogram with Mean & Median Labels
    sns.histplot(df_no_outliers[col], bins=30, kde=True, ax=axes[0, 0], color="blue")
    axes[0, 0].axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
    axes[0, 0].axvline(median_val, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_val:.2f}')
    axes[0, 0].set_title(f"Histogram of {col}", fontsize=14)
    axes[0, 0].set_xlabel(col, fontsize=12)
    axes[0, 0].set_ylabel("Frequency", fontsize=12)
    axes[0, 0].legend()

    ## 2️⃣ Boxplot with Labels
    sns.boxplot(y=df_no_outliers[col], ax=axes[0, 1], color="orange")
    axes[0, 1].set_title(f"Boxplot of {col}", fontsize=14)
    axes[0, 1].set_ylabel(col, fontsize=12)

    # Annotate median value
    axes[0, 1].text(0, median_val, f'Median: {median_val:.2f}', 
                    ha='left', fontsize=10, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7))

    ## 3️⃣ Density Plot (KDE)
    sns.kdeplot(df_no_outliers[col], fill=True, ax=axes[0, 2], color="green")
    axes[0, 2].set_title(f"Density Plot of {col}", fontsize=14)
    axes[0, 2].set_xlabel(col, fontsize=12)
    axes[0, 2].set_ylabel("Density", fontsize=12)

    ## 4️⃣ Violin Plot with Labels
    sns.violinplot(y=df_no_outliers[col], ax=axes[1, 0], color="purple")
    axes[1, 0].set_title(f"Violin Plot of {col}", fontsize=14)
    axes[1, 0].set_ylabel(col, fontsize=12)

    # Annotate min, max values
    axes[1, 0].text(0, df_no_outliers[col].min(), f'Min: {df_no_outliers[col].min():.2f}', 
                    ha='right', fontsize=10, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7))
    axes[1, 0].text(0, df_no_outliers[col].max(), f'Max: {df_no_outliers[col].max():.2f}', 
                    ha='right', fontsize=10, color='black', 
                    bbox=dict(facecolor='white', alpha=0.7))

    ## 5️⃣ Log Transformation for Skewed Data
    if abs(skewness) > 1:  # Apply log transformation for highly skewed data
        log_col = np.log1p(df_no_outliers[col])  # log1p to avoid log(0) issue
        sns.histplot(log_col, bins=30, kde=True, ax=axes[1, 1], color="brown")
        axes[1, 1].set_title(f"Log-Transformed Histogram of {col}", fontsize=14)
        axes[1, 1].set_xlabel(f"Log of {col}", fontsize=12)
        axes[1, 1].set_ylabel("Frequency", fontsize=12)

    ## 6️⃣ Skewness & Kurtosis Annotations
    axes[1, 2].text(0.1, 0.5, f'Skewness: {skewness:.2f}\nKurtosis: {kurtosis:.2f}',
                    fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))
    axes[1, 2].set_title(f"Skewness & Kurtosis for {col}", fontsize=14)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


# Feature Engineering
############
df['Order_Fulfillment_Efficiency'] = df['Inventory_Level'] / (df['Order_Quantity'] + 1)

######### Categorical Features
########
def classify_bottleneck(change_type):
    if change_type == "Supplier Delay":
        return "Supplier Issue"
    elif change_type in ["Feature Update", "Delayed Approval"]:
        return "Production Issue"
    else:
        return "No Issue"

df['Supply_Chain_Bottleneck'] = df['Change_Type'].apply(classify_bottleneck)

####
machine_counts = df['Machine_ID'].value_counts().reset_index()
machine_counts.columns = ['Machine_ID', 'Order_Count']

def categorize_demand(count):
    if count > 50:
        return 'High Demand'
    elif count >= 20:
        return 'Moderate Demand'
    else:
        return 'Low Demand'

machine_counts['Machine_Demand_Level'] = machine_counts['Order_Count'].apply(categorize_demand)
df = df.merge(machine_counts[['Machine_ID', 'Machine_Demand_Level']], on='Machine_ID', how='left')


# Store Data Types
data_types = df.dtypes

######## Change Type
columns_to_convert = [
    'Order_Quantity', 
    'Inventory_Level', 
    'Lead_Time_Days', 
    'Delay_Days', 
    'Order_Fulfillment_Efficiency'
]

df[columns_to_convert] = df[columns_to_convert].astype(int)


# Save
df.to_csv('processed_inventory_data.csv', index=False)



############## EDA After Pre-processing


# ======================= Numeric Columns Analysis =======================

numerical_columns = ['Order_Quantity', 'Inventory_Level', 'Lead_Time_Days', 'Delay_Days']
numerical_summary = df_no_outliers[numerical_columns].describe()  # Summary statistics

data_types = df_no_outliers.dtypes

# ======================= Column-wise Summary for Numeric Data =======================

column_summary = []

for col in numerical_columns:
    # Calculate statistics
    q1 = df_no_outliers[col].quantile(0.25)
    q3 = df_no_outliers[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = df_no_outliers[(df_no_outliers[col] < lower_bound) | (df_no_outliers[col] > upper_bound)][col].count()

    # Append details to the summary
    column_summary.append({
        "Column": col,
        "Data Type": df_no_outliers[col].dtype,
        "Null Count": df_no_outliers[col].isnull().sum(),
        "Not Null Count": df_no_outliers[col].notnull().sum(),
        "Missing Percentage": round(df_no_outliers[col].isnull().sum() / len(df_no_outliers) * 100, 2),
        "Mean": df_no_outliers[col].mean(skipna=True),
        "Median": df_no_outliers[col].median(skipna=True),
        "Std Dev": df_no_outliers[col].std(skipna=True),
        "Min": df_no_outliers[col].min(skipna=True),
        "Max": df_no_outliers[col].max(skipna=True),
        "Skewness": df_no_outliers[col].skew(skipna=True),
        "Kurtosis": df_no_outliers[col].kurtosis(skipna=True),
        "Outliers Count": outliers
    })

# Convert to DataFrame
column_summary_df = pd.DataFrame(column_summary)

# Export to CSV
column_summary_df.to_csv("eda_column_summary_after_preprocessing_outliers.csv", index=False)

print("EDA column summary with outliers, skewness, and kurtosis exported successfully! ✅")



################# CATEGORICAL COLUMN ANALYSIS ##################

# Identify categorical columns
categorical_columns = df_no_outliers.select_dtypes(include=['object', 'category']).columns.tolist()

# Summary storage
categorical_summary = []

# Define high-cardinality threshold
high_cardinality_threshold = 50  

# Total count of rows in the dataset
total_rows = len(df_no_outliers)

for col in categorical_columns:
    unique_values = df_no_outliers[col].nunique()
    mode_value = df_no_outliers[col].mode().iloc[0] if not df_no_outliers[col].mode().empty else None
    missing_count = df_no_outliers[col].isnull().sum()
    not_null_count = df_no_outliers[col].notnull().sum()
    missing_percentage = round((missing_count / total_rows) * 100, 2)
    is_high_cardinality = "Yes" if unique_values > high_cardinality_threshold else "No"

    # Get top 5 frequent values with counts & percentage
    top_values = df_no_outliers[col].value_counts().head(5)
    top_values_dict = top_values.to_dict()
    top_values_percentage = (top_values / total_rows * 100).round(2).to_dict()

    # Store summary
    categorical_summary.append({
        "Column": col,
        "Total Count": total_rows,
        "Unique Values": unique_values,
        "Mode": mode_value,
        "Top 5 Frequent Values": top_values_dict,
        "Top 5 Percentage Distribution": top_values_percentage,
        "Null Count": missing_count,
        "Not Null Count": not_null_count,
        "Missing Percentage": missing_percentage,
        "High Cardinality": is_high_cardinality
    })

# Convert to DataFrame
categorical_summary_df = pd.DataFrame(categorical_summary)

# Export to CSV
categorical_summary_df.to_csv("eda_categorical_after_preprocessing_summary.csv", index=False)

print("EDA categorical column summary exported successfully! ✅")






################## Python To SQL ###################################
import urllib.parse
from sqlalchemy import create_engine

# PostgreSQL connection details
db_name = "Cleaned_Project3"
user = "####"
password = "####"
host = "#####"
port = "5433"

# Encode the password correctly
encoded_password = urllib.parse.quote_plus(password)

# Create a connection to PostgreSQL
engine = create_engine(f"postgresql://{user}:{encoded_password}@{host}:{port}/{db_name}")

# Define table name
table_name = "demand_production_data"

# Load the DataFrame into PostgreSQL
df.to_sql(table_name, engine, if_exists="replace", index=False)

print("✅ Data successfully loaded into PostgreSQL!")
