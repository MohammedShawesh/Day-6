import numpy as np

np.random.seed(42)
##lab 1

# 1.Create a 3×3 NumPy array with numbers from 1 to 9, and print:
arr = np.arange(1,10,1)
arr = arr.reshape((3,3))
#1A.shape, size, ndim, and dtype
print(arr.shape)
print(arr.ndim)
print(arr.size)
print(arr.dtype)
#1B.total bytes in memory (nbytes)
print(arr.nbytes)

#2.Use NumPy built-ins to create:
#2A.a 2×4 array of random numbers
arr2 = np.random.random((2,4))
print(f"Array 2 (2a) {arr2}")
#2B. 3×3 identity matrix
arr3 = np.eye(3,3)
print(f"Array 3 (2b) \n {arr3}")
#2C. an array of 10 values evenly spaced between 0 and 5
arr4 = np.linspace(0,5,5)
print(f"Array 4 (2c) \n {arr4}")

#3.Create a 1D array from 1 to 12, reshape it to (3, 4), then:
arr5 = np.arange(1,13).reshape((3,4))
print(f"Array 5 (3) \n {arr5}")
#3A. Replace the second row with all zeros
arr5 [1,:] = 0
#3B Print the resulting array
print(f"Array 5 (3A) \n {arr5}")

#4 broadcasting to add
base = np.array([[1, 2, 3], [4, 5, 6]])
bonus = np.array([10, 20, 30])
result = base + bonus
print(np.mean(result, axis=0))

#5Given the 2D array:
data = np.array([[3, 6, 9], [2, 4, 8], [1, 5, 7]])
mean = np.mean(data)
standard_deviation = np.std(data)
minimum_value = np.min(data)
print(f"Minimum value : {minimum_value} and standard deviation : {standard_deviation} and mean : {mean}")
sum_row = np.sum(data, axis=1)
sum_col = np.sum(data, axis=0)
print(f"Sum of rows {sum_row}and cols {sum_col}")

##Lab 2
#1.Compute average hourly consumption (per column).
consumption_data = np.random.randint(1, 11, size=(100, 24))
avg_hourly_consumption = consumption_data.mean(axis=0)

#2.Find the house with highest total daily consumption.
total_daily_consumption = consumption_data.sum(axis=1)
highest_consumption_house_index = total_daily_consumption.argmax()
highest_consumption_house_id = highest_consumption_house_index + 1
max_consumption = total_daily_consumption.max()

#3.Detect outliers: houses with total consumption > 95th percentile.
threshold_95 = np.percentile(total_daily_consumption, 95)
outlier_indices = np.where(total_daily_consumption > threshold_95)[0]
outlier_house_ids = outlier_indices + 1
outlier_consumption_values = total_daily_consumption[outlier_indices]

#4.Smooth hourly readings for a single house using a 3-hour moving average.
house_index = 9
house_readings = consumption_data[house_index, :]
window_size = 3
kernel = np.ones(window_size)
smoothed_sum = np.convolve(house_readings, kernel, mode='valid')
smoothed_avg = smoothed_sum / window_size

#5.Normalize each household’s readings so their daily total = 1 (relative profile).
totals_column = total_daily_consumption.reshape(-1, 1)
normalized_profiles = consumption_data / totals_column

# Print Results
print("--- Power Consumption Analysis Results ---")
print(f"Dataset Shape (Houses x Hours): {consumption_data.shape}")
print("-" * 50)

# 1. Average Hourly Consumption
print("1. Average Hourly Consumption (24 values):")
print(avg_hourly_consumption.round(2))
print("-" * 50)

# 2. House with Highest Total Daily Consumption
print("2. Highest Total Daily Consumption:")
print(f"   House ID: {highest_consumption_house_id}")
print(f"   Total Consumption (kWh): {max_consumption}")
print("-" * 50)

# 3. Detect Outliers (> 95th Percentile)
print("3. Outlier Detection (> 95th Percentile):")
print(f"   95th Percentile Threshold: {threshold_95:.2f} kWh")
print(f"   Outlier House IDs: {outlier_house_ids}")
print(f"   Outlier Consumption Values: {outlier_consumption_values}")
print("-" * 50)

# 4. Smoothed Hourly Readings (House 10)
print(f"4. Smoothed Readings for House {highest_consumption_house_id}:")
print(f"   Original Readings (First 5): {house_readings[:5]}")
print(f"   3-Hour Moving Average (First 5): {smoothed_avg[:5].round(2)}")
print("-" * 50)

# 5. Normalized Profiles (Daily Total = 1)
print("5. Normalized Consumption Profiles (Relative Share):")
# Check the sum of the first house to confirm normalization
first_house_normalized_sum = normalized_profiles[0, :].sum()
print(f"   Shape of Normalized Data: {normalized_profiles.shape}")
print(f"   Sum of Normalized Profile for House 1: {first_house_normalized_sum:.4f} (Confirms total = 1)")
print(f"   Normalized Profile (First 5 values of House 1): {normalized_profiles[0, :5].round(4)}")