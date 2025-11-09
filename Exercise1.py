import numpy as np

# 1. Create a python list then convert to numpy array and print shape dtype and ndim
py_list  = [ 10, 20,30 ,40 ,50]

arr1 = np.array(py_list)
print(arr1.shape)
print(arr1.dtype)
print(arr1.ndim)
#2. print array 3*3 array of ones
print(np.ones((3,3)))
#print 4*1 array filled with value 7
print(np.full((4,1), 7))
#print array of even numbers from2 to20
print(np.arange(2,20,2))

#3.given array a and b due multi , sub , and add
a = np.array([2,4,6,8])
b = np.array([1,3,5,7])
print(a+b)
print(a-b)
print(a*b)