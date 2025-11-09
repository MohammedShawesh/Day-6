import numpy as np

np.random.seed(42)
temp = np.random.randint(7,40,365)
humidity = np.random.randint(0,100,365)
wind_speed = np.random.randint(10,220,365)
stacked_data = np.column_stack([temp, humidity, wind_speed])

#1.Compute the monthly average temperature (assuming 30 days per month).
temp_360 = temp[:360]
monthly_temp = temp_360.reshape(12, 30)
monthly_avg_temp = monthly_temp.mean(axis=1)
print(monthly_avg_temp.round(1))

#2.
window_size = 7
kernel = np.ones(window_size)
rolling_sum = np.convolve(temp, kernel, mode='valid')
rolling_avg = rolling_sum / window_size
max_avg_temp = rolling_avg.max()
start_index = rolling_avg.argmax()
start_day = start_index + 1
end_day = start_index + window_size
hottest_week_temps = temp[start_index:end_day]


print(f"Hottest Week (Days): Day {start_day} through Day {end_day}.")
print(f"Maximum Average: ${max_avg_temp:.2f}\\degree\\text{{C}}.")
print(f"Temperatures: The specific temperatures during this 7-day period are {hottest_week_temps}.")