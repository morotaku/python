#### Hollow Gaussian beam
from math import *
import numpy as np
from scipy.special import *
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.colors import LinearSegmentedColormap

print(factorial(170))
#Parameters(μm)
w=1000 #Beam waist
lam=0.64 #Wave length
k=2*np.pi/lam #Wave number
zr=w**2*k/2 #Rayleigh length

z=10 #Distance from focus position 
R=z+zr**2/z #Beam radius
W=w*(1+(z/zr)**2)**0.5 #Beam size

f=10**5 #Focus length
s=0#200000 #Distance from the input plane to lens
G0=1
#Z=z/zr

#ABCD matrix
A=-z/f
B=-(z/f)*s+f+z
C=-1/f
D=1-s/f
print(A)#,B,C,D)
#x-y　coordinate
N=200
L=50 #Display range
X=np.linspace(-L,L,N)
Y=np.linspace(-L,L,N)
x,y=np.meshgrid(X,Y)
r=np.sqrt(x**2+y**2)


i=1j



#Laguerre polynominal
def Lag(l,x):
    L=[0]*(l+1)
    for i in range(0,l+1):
        if i==0:
            L[i]=1
        elif i==1:
            L[i]=-x+1
        else:
            L[i]=(2*(i-1)+1-x)*L[i-1]-(i-1)**2*L[i-2]
    return L[l]
a=((k*r/(2*B))**2)

b=(1/w**2)
c=(i*k*A/(2*B))

#print(Lag(3,a/(b+c)))
array1 = np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]])

array2 = np.array([[1,2,3],
                 [4,5,6],
                 [7,8,9]])

