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

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming final_data is your original DataFrame with categorical variables
# Convert categorical well-being indicators to numeric values (for example, using .cat.codes)
well_being_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs',
                      'Dealpr', 'Thcklr', 'Goodme', 'Clsep',
                      'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

for col in well_being_columns:
    final_data[col] = final_data[col].cat.codes

# Now calculate the Well_Being_Score
final_data['Well_Being_Score'] = final_data[well_being_columns].sum(axis=1)

# Prepare the data: select features and target variable
features = final_data.drop(columns=well_being_columns + ['Well_Being_Score'])

target = final_data['Well_Being_Score']

final_data.head()

final_data.info()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming final_data is your original DataFrame with categorical variables
# Convert categorical well-being indicators to numeric values (for example, using .cat.codes)
well_being_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs',
                      'Dealpr', 'Thcklr', 'Goodme', 'Clsep',
                      'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

for col in well_being_columns:
    final_data[col] = final_data[col].cat.codes

# Now calculate the Well_Being_Score
final_data['Well_Being_Score'] = final_data[well_being_columns].sum(axis=1)

# Prepare the data: select features and target variable
features = final_data.drop(columns=well_being_columns + ['Well_Being_Score'])

target = final_data['Well_Being_Score']

final_data.head()

final_data.info()

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming final_data is your original DataFrame with categorical variables
# Convert categorical well-being indicators to numeric values (for example, using .cat.codes)
well_being_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs',
                      'Dealpr', 'Thcklr', 'Goodme', 'Clsep',
                      'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

for col in well_being_columns:
    final_data[col] = final_data[col].cat.codes

# Now calculate the Well_Being_Score
final_data['Well_Being_Score'] = final_data[well_being_columns].sum(axis=1)

# Prepare the data: select features and target variable
features = final_data.drop(columns=well_being_columns + ['Well_Being_Score'])

target = final_data['Well_Being_Score']

final_data.head()

final_data.info()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming final_data is your original DataFrame with categorical variables
# Convert categorical well-being indicators to numeric values (for example, using .cat.codes)
well_being_columns = ['Optm', 'Usef', 'Relx', 'Intp', 'Engs',
                      'Dealpr', 'Thcklr', 'Goodme', 'Clsep',
                      'Conf', 'Mkmind', 'Loved', 'Intthg', 'Cheer']

for col in well_being_columns:
    final_data[col] = final_data[col].cat.codes

# Now calculate the Well_Being_Score
final_data['Well_Being_Score'] = final_data[well_being_columns].sum(axis=1)

# Prepare the data: select features and target variable
features = final_data.drop(columns=well_being_columns + ['Well_Being_Score'])

target = final_data['Well_Being_Score']

final_data.head()

final_data.info()

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Apply Polynomial Features
poly = PolynomialFeatures(degree=2)  # Try different degrees (e.g., 2 or 3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the model
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train_poly, y_train)

# Make predictions
y_train_pred = poly_reg_model.predict(X_train_poly)
y_test_pred = poly_reg_model.predict(X_test_poly)

# Evaluate the model
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"R² Score: {r2}")

from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

# Evaluate the model
train_rmse_lr = mean_squared_error(y_train, y_train_pred_lr, squared=False)
test_rmse_lr = mean_squared_error(y_test, y_test_pred_lr, squared=False)
r2_lr = r2_score(y_test, y_test_pred_lr)

print(f"Linear Regression Train RMSE: {train_rmse_lr}")
print(f"Linear Regression Test RMSE: {test_rmse_lr}")
print(f"Linear Regression R² Score: {r2_lr}")

from sklearn.ensemble import GradientBoostingRegressor

# Initialize the model
gb_model = GradientBoostingRegressor(random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions
y_train_pred = gb_model.predict(X_train)
y_test_pred = gb_model.predict(X_test)

# Evaluate the model
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"R² Score: {r2}")

from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Evaluate the model
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, y_test_pred, squared=False)
r2 = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse}")
print(f"Test RMSE: {test_rmse}")
print(f"R² Score: {r2}")

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Select numerical features to scale
numerical_columns = ['C_we', 'C_wk', 'G_we', 'G_wk', 'S_we', 'S_wk', 'T_we', 'T_wk', 'Total_Screen_Time','Well_Being_Score']

# Initialize the scaler
scaler = StandardScaler()

# Scale the numerical features
final_data_scaled = final_data.copy()  # Copy the original data
final_data_scaled[numerical_columns] = scaler.fit_transform(final_data[numerical_columns])

# Separate features and target
features_scaled = final_data_scaled.drop(columns=['Well_Being_Score'])
target_scaled = final_data_scaled['Well_Being_Score']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target_scaled, test_size=0.2, random_state=42)

from sklearn.linear_model import LinearRegression

# Initialize the Linear Regression model
lr_model = LinearRegression()

# Train the model
lr_model.fit(X_train, y_train)

# Make predictions
y_train_pred_lr = lr_model.predict(X_train)
y_test_pred_lr = lr_model.predict(X_test)

# Evaluate the model
train_rmse_lr = mean_squared_error(y_train, y_train_pred_lr, squared=False)
test_rmse_lr = mean_squared_error(y_test, y_test_pred_lr, squared=False)
r2_lr = r2_score(y_test, y_test_pred_lr)

print(f"Linear Regression Train RMSE: {train_rmse_lr}")
print(f"Linear Regression Test RMSE: {test_rmse_lr}")
print(f"Linear Regression R² Score: {r2_lr}")

# Apply Polynomial Features
poly = PolynomialFeatures(degree=2)  # Try different degrees (e.g., 2 or 3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train the model
poly_reg_model = LinearRegression()
poly_reg_model.fit(X_train_poly, y_train)

# Make predictions
y_train_pred_ = poly_reg_model.predict(X_train_poly)
y_test_pred = poly_reg_model.predict(X_test_poly)

# Evaluate the model
train_rmse_pol = mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse_pol = mean_squared_error(y_test, y_test_pred, squared=False)
r2_pol = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse_pol}")
print(f"Test RMSE: {test_rmse_pol}")
print(f"R² Score: {r2_pol}")


from sklearn.ensemble import GradientBoostingRegressor

# Initialize the Gradient Boosting model
gb_model = GradientBoostingRegressor(random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions
y_train_pred_gb = gb_model.predict(X_train)
y_test_pred_gb = gb_model.predict(X_test)

# Evaluate the model
train_rmse_gb = mean_squared_error(y_train, y_train_pred_gb, squared=False)
test_rmse_gb = mean_squared_error(y_test, y_test_pred_gb, squared=False)
r2_gb = r2_score(y_test, y_test_pred_gb)

print(f"Train RMSE: {train_rmse_gb}")
print(f"Test RMSE: {test_rmse_gb}")
print(f"R² Score: {r2_gb}")

from sklearn.ensemble import RandomForestRegressor

# Initialize the Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_train_pred_rf = rf_model.predict(X_train)
y_test_pred_rf = rf_model.predict(X_test)

# Evaluate the model
train_rmse_rf = mean_squared_error(y_train, y_train_pred_rf, squared=False)
test_rmse_rf = mean_squared_error(y_test, y_test_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_test_pred)

print(f"Train RMSE: {train_rmse_rf}")
print(f"Test RMSE: {test_rmse_rf}")
print(f"R² Score: {r2_rf}")

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# Define a function to calculate NRMSE
def nrmse(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return rmse / (np.max(y_true) - np.min(y_true))

# Initialize DataFrame to store results
results = pd.DataFrame(columns=['Model', 'MAE', 'MSE', 'RMSE', 'NRMSE', 'R² Score'])

# --- Linear Regression ---
mae_lr = mean_absolute_error(y_test, y_test_pred_lr)
mse_lr = mean_squared_error(y_test, y_test_pred_lr)
rmse_lr = mean_squared_error(y_test, y_test_pred_lr, squared=False)
nrmse_lr = nrmse(y_test, y_test_pred_lr)
r2_lr = r2_score(y_test, y_test_pred_lr)

# Add the Linear Regression results to the DataFrame
results = pd.concat([results, pd.DataFrame({
    'Model': ['Linear Regression'], 
    'MAE': [mae_lr], 
    'MSE': [mse_lr], 
    'RMSE': [rmse_lr], 
    'NRMSE': [nrmse_lr], 
    'R² Score': [r2_lr]
})], ignore_index=True)

# --- Polynomial Regression ---
mae_pol = mean_absolute_error(y_test, y_test_pred)
mse_pol = mean_squared_error(y_test, y_test_pred)
rmse_pol = mean_squared_error(y_test, y_test_pred, squared=False)
nrmse_pol = nrmse(y_test, y_test_pred)
r2_pol = r2_score(y_test, y_test_pred)

# Add the Polynomial Regression results to the DataFrame
results = pd.concat([results, pd.DataFrame({
    'Model': ['Polynomial Regression'], 
    'MAE': [mae_pol], 
    'MSE': [mse_pol], 
    'RMSE': [rmse_pol], 
    'NRMSE': [nrmse_pol], 
    'R² Score': [r2_pol]
})], ignore_index=True)

# --- Gradient Boosting ---
mae_gb = mean_absolute_error(y_test, y_test_pred_gb)
mse_gb = mean_squared_error(y_test, y_test_pred_gb)
rmse_gb = mean_squared_error(y_test, y_test_pred_gb, squared=False)
nrmse_gb = nrmse(y_test, y_test_pred_gb)
r2_gb = r2_score(y_test, y_test_pred_gb)

# Add the Gradient Boosting results to the DataFrame
results = pd.concat([results, pd.DataFrame({
    'Model': ['Gradient Boosting'], 
    'MAE': [mae_gb], 
    'MSE': [mse_gb], 
    'RMSE': [rmse_gb], 
    'NRMSE': [nrmse_gb], 
    'R² Score': [r2_gb]
})], ignore_index=True)

# --- Random Forest ---
mae_rf = mean_absolute_error(y_test, y_test_pred_rf)
mse_rf = mean_squared_error(y_test, y_test_pred_rf)
rmse_rf = mean_squared_error(y_test, y_test_pred_rf, squared=False)
nrmse_rf = nrmse(y_test, y_test_pred_rf)
r2_rf = r2_score(y_test, y_test_pred_rf)

# Add the Random Forest results to the DataFrame
results = pd.concat([results, pd.DataFrame({
    'Model': ['Random Forest'], 
    'MAE': [mae_rf], 
    'MSE': [mse_rf], 
    'RMSE': [rmse_rf], 
    'NRMSE': [nrmse_rf], 
    'R² Score': [r2_rf]
})], ignore_index=True)

# Display the results in a table format
print(results)
import matplotlib.pyplot as plt
import seaborn as sns

# Set the figure size
plt.figure(figsize=(18, 8))

# Create subplots for MAE, MSE, RMSE, NRMSE, and R² Score
fig, axes = plt.subplots(3, 2, figsize=(14, 12))

# Plot MAE
sns.barplot(x='Model', y='MAE', data=results, ax=axes[0, 0], palette='viridis')
axes[0, 0].set_title('MAE Comparison')

# Plot MSE
sns.barplot(x='Model', y='MSE', data=results, ax=axes[0, 1], palette='plasma')
axes[0, 1].set_title('MSE Comparison')

# Plot RMSE
sns.barplot(x='Model', y='RMSE', data=results, ax=axes[1, 0], palette='inferno')
axes[1, 0].set_title('RMSE Comparison')

# Plot NRMSE
sns.barplot(x='Model', y='NRMSE', data=results, ax=axes[1, 1], palette='cividis')
axes[1, 1].set_title('NRMSE Comparison')

# Plot R² Score
sns.barplot(x='Model', y='R² Score', data=results, ax=axes[2, 0], palette='magma')
axes[2, 0].set_title('R² Score Comparison')

# Remove the empty subplot (axes[2, 1])
fig.delaxes(axes[2, 1])

# Rotate x labels for better readability
for ax in axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.show()




