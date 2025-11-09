import numpy as np

latency = np.random.uniform(5 , 500, 100)
throughput = np.random.uniform(0 , 50, 100)
retransmission =np.random.uniform(0,20,100)
print(latency)
print(throughput)
print(retransmission)
stacked_data = np.column_stack([latency, throughput, retransmission])
print(stacked_data)
#1 Print the shape, dtype, and first 5 rows of data.
print(stacked_data.shape)
print(stacked_data.dtype)
print(stacked_data[:5])
#2 Compute the average latency, minimum throughput, and maximum retransmission.
avg_latency = stacked_data[:, 0].mean()
min_throughput = stacked_data[:, 1].min()
max_retransmission = stacked_data[:, 2].max()

#3.Find customers whose latency > mean + 2×std (potential issues).
mean_latency = latency.mean()
std_latency = latency.std()
threshold = mean_latency + 2 * std_latency
potential_issues = np.where(latency >= threshold)[0]
print(potential_issues)

#4.Compute the correlation between latency and throughput.
correlation = np.corrcoef(latency, throughput)
latency_throughput_correlation = correlation[0, 1]
print(round(latency_throughput_correlation, 4))

#5.Normalize all features (Z-score).
dMean = stacked_data.mean(axis=0)
dstd = stacked_data.std(axis=0)

normalized_latency = (stacked_data[:,0] - stacked_data[:,0].mean() ) / stacked_data[:,0].std()
normalized_throughput = (stacked_data[:,1] - stacked_data[:,1].mean() ) / stacked_data[:,1].std()
normalized_retransmission = (stacked_data[:,2] - stacked_data[:,2].mean() ) / stacked_data[:,2].std()
print(f"the normalized latency is {normalized_latency} \n the normalized throughput is {normalized_throughput} \n the normalized retransmission is {normalized_retransmission}")