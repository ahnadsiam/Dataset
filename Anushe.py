import pandas as pd
import matplotlib.pyplot as plt

# Load the datasets
df = pd.read_csv('dataset1 (1).csv')
df = pd.read_csv('dataset2 (1).csv')
df = pd.read_csv('dataset3 (1).csv')

# Reading the CSV files
dataset1 = pd.read_csv('dataset1 (1).csv')
dataset2 = pd.read_csv('dataset2 (1).csv')
dataset3 = pd.read_csv('dataset3 (1).csv')

# Merge the datasets on the 'ID' column
merged_data = dataset1.merge(dataset2, on='ID', how='inner').merge(dataset3, on='ID', how='inner')

# Calculate the median for each numeric column in the merged dataset
medians = merged_data.median()
print("Medians:\n", medians)

# Plot a bar chart for median screen time variables
screen_time_cols = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk']
median_screen_time = merged_data[screen_time_cols].median()

plt.figure(figsize=(10, 6))
median_screen_time.plot(kind='bar', color='skyblue')
plt.title('Median Screen Time (Hours) on Weekdays and Weekends')
plt.ylabel('Hours')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plot a bar chart for median well-being indicators
well_being_cols = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']
median_well_being = merged_data[well_being_cols].median()

plt.figure(figsize=(10, 6))
median_well_being.plot(kind='bar', color='lightgreen')
plt.title('Median Well-Being Indicators')
plt.ylabel('Well-Being Score (1-5)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
