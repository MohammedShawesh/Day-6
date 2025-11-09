import numpy as np

arr = np.arange(10,20)
print(arr[3])
print(arr[2:7])
print(arr[::2])

arr2 = np.arange(1,13)
arr2 = np.reshape(arr2, (3,4))
print(arr2.shape)
arr2 = arr2.reshape(-1)
print(arr2.shape)

M = np.array([[5, 10, 15], [20, 25, 30]])
v = np.array([1, 2, 3])
result = M+v
mean = np.mean(result)
summ = np.sum(result)
maxx = np.max(result)
print(result)
print(f"the sum is {summ} the max is {maxx} the mean is {mean}")