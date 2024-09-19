import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
file_path = '/mnt/data/dataset1 (1).csv'
dataset= pd.read_csv('dataset1 (1).csv')

# Calculate mode for 'gender', 'minority', and 'deprived' columns
mode_gender = dataset['gender'].mode()[0]
mode_minority = dataset['minority'].mode()[0]
mode_deprived = dataset['deprived'].mode()[0]

# Output the mode for each column
modes = {
    "Mode": {
        "gender": mode_gender,
        "minority": mode_minority,
        "deprived": mode_deprived
    }
}
print(modes)

# Visualize the data
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Gender visualization
dataset['gender'].value_counts().plot(kind='bar', ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Gender Distribution')
axes[0].set_xticklabels(['Female (0)', 'Male (1)'], rotation=0)
axes[0].set_ylabel('Count')

# Minority visualization
dataset['minority'].value_counts().plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='black')
axes[1].set_title('Minority Group Distribution')
axes[1].set_xticklabels(['Majority (0)', 'Minority (1)'], rotation=0)
axes[1].set_ylabel('Count')

# Deprived visualization
dataset['deprived'].value_counts().plot(kind='bar', ax=axes[2], color='lightcoral', edgecolor='black')
axes[2].set_title('Deprived Area Distribution')
axes[2].set_xticklabels(['Not Deprived (0)', 'Deprived (1)'], rotation=0)
axes[2].set_ylabel('Count')

# Show plot
plt.tight_layout()
plt.show()
