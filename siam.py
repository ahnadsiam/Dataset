import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np
# Load the datasets
df = pd.read_csv('dataset2 (1).csv')
df = pd.read_csv('dataset3 (1).csv')

# Reading the CSV files into DataFrames
df_screen_time = pd.read_csv('dataset2 (1).csv')
df_well_being= pd.read_csv('dataset3 (1).csv')



# Display the first few rows of the datasets to ensure the data is loaded correctly
print("\nScreen Time Data: ")
print(df_screen_time.head())

print("\nWell-Being Data: ")
print(df_well_being.head())

# Merge Dataset 2 and Dataset 3 on 'ID'
merged_data = pd.merge(df_screen_time, df_well_being, on='ID')

# Calculate total screen time (sum across all devices for both weekdays and weekends)
merged_data['total_screen_time'] = (
    merged_data['C_we'] + merged_data['C_wk'] +
    merged_data['G_we'] + merged_data['G_wk'] +
    merged_data['S_we'] + merged_data['S_wk'] +
    merged_data['T_we'] + merged_data['T_wk']
)

# Split data into two groups based on the median of total screen time
median_screen_time = merged_data['total_screen_time'].median()
high_screen_time = merged_data[merged_data['total_screen_time'] > median_screen_time]
low_screen_time = merged_data[merged_data['total_screen_time'] <= median_screen_time]

# List of well-being indicators
well_being_indicators = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs', 'Dealpr', 
                         'Thcklr', 'Goodme', 'Clsep', 'Conf', 'Mkmind', 
                         'Loved', 'Intthg', 'Cheer']

# List to store the results from the hypothesis testing
results = []
improved_count = 0  # To count how many well-being indicators show improvement

# Loop through each well-being indicator and perform the t-test
for var in well_being_indicators:
    # Get the means for both high and low screen time groups
    mean_high_screen_time = high_screen_time[var].mean()
    mean_low_screen_time = low_screen_time[var].mean()
    
    # Perform the t-test
    t_stat, p_value = st.ttest_ind(high_screen_time[var], low_screen_time[var], equal_var=False)
    
    # Determine if the p-value is less than 0.05 (significance level)
    if p_value < 0.05:
        conclusion = f"Reducing screen time likely improves {var}"
        improved_count += 1  # Increment the count for improvements
    else:
        conclusion = f"No significant effect of screen time on {var}"
    
    # Save the results
    results.append({
        'Variable': var,
        'Mean High': mean_high_screen_time,
        'Mean Low': mean_low_screen_time,
        'T-statistic': t_stat,
        'P-Value': p_value,
        'Conclusion': conclusion
    })

# Convert results to a DataFrame for easy viewing
results_df = pd.DataFrame(results)

# Display the results
print(results_df)

# Bar chart visualization to show the number of well-being indicators that improved
labels = ['Improved Well-Being', 'No Improvement']
sizes = [improved_count, len(well_being_indicators) - improved_count]

# Create a bar chart
fig, ax = plt.subplots()
ax.bar(labels, sizes, color=['#66b3ff', '#ff9999'])

# Adding labels and title
ax.set_ylabel('Number of Well-Being Indicators')
ax.set_title('Effect of Reducing Screen Time on Well-Being Indicators')

# Display the bar chart
plt.show()

