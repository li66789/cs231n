import numpy as np
# a = np.array([[1,2,3],[2,3,4],[4,5,6]])
# print(a[:,1].shape,a[:,1])
# b = np.array([[1,2,3]])
# a[:,1]+=b[0]
# print(a[:,1].shape,a[:,1],"\n",a)
# print(a[1].shape,a[1],b.shape)
# # print(c)
str = np.array([[1,2,3],[5,6,7]])
a = str
b = a
# b[0] = 33
print(str.max(axis=1))
print(str.max(axis=0))