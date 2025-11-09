import numpy as np

np.random.seed(1)
transactions = np.random.randint(10, 500, size=(10, 7))
#1. Compute each customer’s total and average weekly spending.
print(transactions.mean(axis=1))

#2.Find the day with the highest total spending across all customers.
sum_per_day =transactions.sum(axis=0)
print(sum_per_day)
print(sum_per_day.max())
print(sum_per_day.argmax()+1)

#3. Identify customers whose spending variance > 10000 (unstable behavior).
a3 = np.var(transactions, axis = 1)

print(np.where(a3 > 10000))

#4Standardize the dataset by dividing each column by its mean.
column_means = np.mean(transactions, axis=0)
standardized_dataset = transactions / column_means

#5.Find the top 3 customers with the highest weekly spending total.
total_weekly_spending = transactions.sum(axis=1)
top_3_indices = np.argsort(total_weekly_spending)[::-1][:3]
