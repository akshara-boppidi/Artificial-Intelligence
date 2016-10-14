import math
import matplotlib.pyplot as plt
import numpy as np
d_vc=10
delta=0.05
e=0.05
MyArray=[]
a = (8/math.pow(e,2))
#print a
n=1000
num_of_iterations =0
for i in range(1,10):
    b = math.log(4* (math.pow(2*n,d_vc)+1)/delta)
    N= a*b
    MyArray.append(N)
    print "n=",n
    round(N)== n
    num_of_iterations = num_of_iterations+1
    print "N=",N
    print
    if():
        break
    else:
        n = round(N)
        continue
print "Number of Iterations=",num_of_iterations
print MyArray
plt.plot(MyArray)
plt.title("Plotting values of N which converges to a steady value of N")
plt.xlabel("Number of Iterations")
plt.ylabel("Values of N")
plt.show(MyArray)