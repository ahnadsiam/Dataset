import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Load the datasets
df = pd.read_csv('dataset1 (1).csv')
df = pd.read_csv('dataset2 (1).csv')

# Reading the CSV files
dataset1 = pd.read_csv('dataset1 (1).csv')
dataset2 = pd.read_csv('dataset2 (1).csv')

# Merge the datasets on the 'ID' column, which is common in both datasets
merged_data = pd.merge(dataset1, dataset2, on='ID')

# 1. Confidence Interval Calculation
# Create a new column for total screen time on weekends (C_we, G_we, S_we, T_we)
merged_data['total_screen_time_we'] = merged_data[['C_we', 'G_we', 'S_we', 'T_we']].sum(axis=1)

# Calculate the mean and standard error of the total screen time on weekends
mean_total_we = np.mean(merged_data['total_screen_time_we'])
std_err_total_we = stats.sem(merged_data['total_screen_time_we'])

# Set the confidence level (95% confidence interval)
confidence_level = 0.95
df = len(merged_data) - 1  # degrees of freedom

# Calculate the confidence interval
confidence_interval = stats.t.interval(confidence_level, df, loc=mean_total_we, scale=std_err_total_we)

# Print the result with a conclusion
print(f"The 95% confidence interval for the total screen time on weekends is between {confidence_interval[0]:.2f} and {confidence_interval[1]:.2f} hours.")
print(f"This means we are 95% confident that the true average total screen time on weekends lies within this interval.")

# 2. Line Plot Visualization
# Create a new column for total screen time on weekdays (C_wk, G_wk, S_wk, T_wk)
merged_data['total_screen_time_wk'] = merged_data[['C_wk', 'G_wk', 'S_wk', 'T_wk']].sum(axis=1)

# Group the data by total weekend screen time and calculate the mean weekday screen time
grouped_data = merged_data.groupby('total_screen_time_we')['total_screen_time_wk'].mean()

# Plot a line plot to show the relationship between weekend and weekday screen time
plt.figure(figsize=(10, 6))
plt.plot(grouped_data.index, grouped_data.values, marker='o', linestyle='-', color='orange')
plt.title('Line Plot: Average Weekday Screen Time by Weekend Screen Time', fontsize=16)
plt.xlabel('Total Screen Time on Weekends (Hours)', fontsize=12)
plt.ylabel('Average Screen Time on Weekdays (Hours)', fontsize=12)
plt.grid(True)
plt.show()
