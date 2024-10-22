import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Suppress FutureWarning for cleaner output
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None
import numpy as np
# Load the datasets
df = pd.read_csv('dataset1 (1).csv')
df = pd.read_csv('dataset2 (1).csv')
df = pd.read_csv('dataset3 (1).csv')

# Reading the CSV files
dataset1 = pd.read_csv('dataset1 (1).csv')
dataset2 = pd.read_csv('dataset2 (1).csv')
dataset3 = pd.read_csv('dataset3 (1).csv')
# List of well-being indicator columns
well_being_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme',
                      'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

# Convert them to categorical (factor-like)
for col in well_being_columns:
    dataset3[col] = dataset3[col].astype('category')

# Merge dataset1 with dataset2 on 'ID'
merged_data = pd.merge(dataset1, dataset2, on='ID')

# Merge the result with dataset3 on 'ID'
final_data = pd.merge(merged_data, dataset3, on='ID')
# Check for missing values
missing_values = final_data.isnull().sum()

print("Missing Values:")
print(missing_values)

# Convert 'gender', 'minority', and 'deprived' to categorical variables
final_data['gender'] = final_data['gender'].astype('category')
final_data['minority'] = final_data['minority'].astype('category')
final_data['deprived'] = final_data['deprived'].astype('category')

# Review the first few rows of the final dataset
final_data.head()

# Remove the ID column from the DataFrame
final_data = final_data.drop(columns=['ID'])

# Display the updated DataFrame information
final_data.info()

# Select numerical columns
numerical_columns = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = final_data[numerical_columns].quantile(0.25)
Q3 = final_data[numerical_columns].quantile(0.75)

# Calculate IQR (Interquartile Range)
IQR = Q3 - Q1

# Define outliers as any value outside 1.5*IQR
outliers = (final_data[numerical_columns] < (Q1 - 1.5 * IQR)) | (final_data[numerical_columns] > (Q3 + 1.5 * IQR))

# Check which rows have outliers in any column
outlier_rows = outliers.any(axis=1)

# Display the rows with outliers
print(f"Number of rows with outliers: {outlier_rows.sum()}")
print(final_data[outlier_rows])
# Remove the rows with outliers
final_data= final_data[~outlier_rows]

# Display the shape of the cleaned dataset
print("Shape of dataset after removing outliers:", final_data.shape)

# Summary statistics for numeric columns (screen time)
numeric_summary = final_data.describe().transpose()
numeric_summary

# Frequency table for categorical well-being indicators
well_being_summary = final_data[well_being_columns].apply(lambda x: x.value_counts()).transpose()
well_being_summary

# Frequency table for categorical s basic demographic information
demographic_columns = ['gender', 'minority', 'deprived']
demographic_information_summary = final_data[demographic_columns].apply(lambda x: x.value_counts()).transpose()
demographic_information_summary

# Assuming final_data is already loaded as a DataFrame

# Summary Statistics of Numeric Screen Time Variables
screen_time_columns = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']  # Update as per your actual column names
numeric_summary = final_data[screen_time_columns].describe().transpose()

# Correlation Between Screen Time and Well-Being Indicators
correlation_summary = final_data[screen_time_columns + well_being_columns].corr()

# Summary of Well-Being Scores Across Different Age Groups (if age column exists)
# Assuming there is an 'age' column
average_well_being_by_age = pd.DataFrame()
if 'age' in final_data.columns:
    age_groups = pd.cut(final_data['age'], bins=[18, 24, 34, 44, 54, 100], labels=['18-24', '25-34', '35-44', '45-54', '55 and older'])
    average_well_being_by_age = final_data.groupby(age_groups)[well_being_columns].mean(numeric_only=True)

# Screen Time Distribution by Gender
screen_time_distribution_by_gender = final_data.groupby('gender')[screen_time_columns].agg(['mean', 'median', 'std'])

# Screen Time Distribution by Deprivation Status
screen_time_distribution_by_deprivation = final_data.groupby('deprived')[screen_time_columns].agg(['mean', 'median', 'std'])

# Count of Responses by Well-Being Indicator and Gender
well_being_count_by_gender = final_data.groupby('gender')[well_being_columns].count()

# Function to display a table with box-like formatting
def display_table(title, df):
    title_length = len(title)
    print(f"\n{'=' * title_length}")
    print(title)
    print(f"{'=' * title_length}\n")
    print(df.to_string(index=True))  # Print the DataFrame as a string for better formatting
    print("\n" + "=" * title_length)

# Display each summary in table format
display_table("Numeric Summary of Screen Time", numeric_summary)

display_table("Correlation Summary Between Screen Time and Well-Being Indicators", correlation_summary)

if not average_well_being_by_age.empty:
    display_table("Average Well-Being by Age Group", average_well_being_by_age)

display_table("Screen Time Distribution by Gender", screen_time_distribution_by_gender)

display_table("Screen Time Distribution by Deprivation Status", screen_time_distribution_by_deprivation)

display_table("Well-Being Count by Gender", well_being_count_by_gender)

# Violin plot of weekend TV usage vs feeling relaxed
plt.figure(figsize=(10, 6))
sns.violinplot(x=final_data['Relx'], y=final_data['T_we'], palette='coolwarm')
plt.title('Weekend TV Usage vs Feeling Relaxed')
plt.xlabel('Feeling Relaxed (Score)')
plt.ylabel('TV Usage on Weekends (Hours)')
plt.show()


# Bar plot of average computer usage (weekdays) by optimism
avg_comp_usage = final_data.groupby('Optm')['C_wk'].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.barplot(x='Optm', y='C_wk', data=avg_comp_usage, palette='coolwarm')
plt.title('Average Computer Usage (Weekdays) by Optimism Score')
plt.xlabel('Optimism Score')
plt.ylabel('Average Computer Usage (Weekdays in Hours)')
plt.show()

# Box plot of gaming usage (weekdays) by deprivation status
plt.figure(figsize=(10, 6))
sns.boxplot(x=final_data['deprived'], y=final_data['G_wk'], palette='Set3')
plt.title('Gaming Usage (Weekdays) by Deprivation Status')
plt.xlabel('Deprivation Status (1=Deprived, 0=Not Deprived)')
plt.ylabel('Gaming Usage on Weekdays (Hours)')
plt.show()

# Line plot of total screen time by well-being scores
final_data['Total_Screen_Time'] = final_data[['S_wk', 'T_wk', 'G_wk', 'C_wk']].sum(axis=1)

plt.figure(figsize=(10, 6))
sns.lineplot(x=final_data['Optm'], y=final_data['Total_Screen_Time'], marker='o')
plt.title('Total Screen Time vs Optimism Score')
plt.xlabel('Optimism Score')
plt.ylabel('Total Screen Time (Weekdays in Hours)')
plt.show()

# Heatmap of average screen time by well-being scores
avg_screen_time = final_data.groupby('Optm')[['S_wk', 'T_wk', 'G_wk', 'C_wk']].mean()

plt.figure(figsize=(10, 6))
sns.heatmap(avg_screen_time, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Average Screen Time by Optimism Score')
plt.xlabel('Screen Time Activity')
plt.ylabel('Optimism Score')
plt.show()

final_data.info()


