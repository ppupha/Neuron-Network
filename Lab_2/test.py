import numpy as np

def mymult(M, A):
    res = [ele * i  for ele, i in zip(M, A)]
    return np.array(res)
	
	
	
	
M = np.array([[1,2],[3,4]])
A = np.array([2,3])

R =  np.array([[1],[2]])
print(M / R)


	
