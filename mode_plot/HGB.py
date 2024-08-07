#### Hollow Gaussian beam
from math import *
import numpy as np
from scipy.special import *
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.colors import LinearSegmentedColormap

#Parameters(μm)
w=1000 #Beam waist
lam=0.64 #Wave length
k=2*np.pi/lam #Wave number
zr=w**2*k/2 #Rayleigh length
print(zr)
z=float(input('position: ')) #Distance from focus position 
R=z+zr**2/z #Beam radius
W=w*(1+(z/zr)**2)**0.5 #Beam size
print(W)
f=10**5 #Focus length
s=0#200000 #Distance from the input plane to lens
G0=1
#Z=z/zr

#ABCD matrix
A=-z/f
B=-(z/f)*s+f+z
C=-1/f
D=1-s/f

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
print(Lag(3,((k*r/(2*B))**2)/((1/w**2)+(i*k*A/(2*B)))))
#Hollow Gaussian Mode
def HGMode(n,r):
    C1=(i*k*G0*factorial(n)/(2*B*w**(2*n)))*((1/(w**2)+(i*k*A/(2*B)))**(-1-n))*np.exp(-i*k*z)
    C2=np.exp((-i*k*D*r**2)/(2*B))
    C3=np.exp(-((k*r/(2*B))**2)/((1/w**2)+(i*k*A/(2*B))))
    lag=Lag(n,((k*r/(2*B))**2)/((1/w**2)+(i*k*A/(2*B))))
    hgm=C1*C2*C3*lag
    HGM=np.real(hgm*hgm.conjugate()) #(E×E^*)
    HGM_n=HGM/(np.amax(HGM)) #Normalization
    return HGM_n

#Plot
rb = LinearSegmentedColormap.from_list('name', ['black', 'white'])
fig = plt.figure(figsize=(8,4))

def Modefig(n):
    ax1 = fig.add_subplot(121)
    #ax1.set_axis_off() #軸off
    im1=plt.imshow(HGMode(n,r),vmax=1,vmin=0,cmap='hot')
    return im1

n=3
print(HGMode(n,r))
Modefig(n)

def Intensity(n):
    ax2=fig.add_subplot(122)
    im2=plt.plot(HGMode(n,r)[100]/np.amax(HGMode(n,r)[100]))
    return im2

Intensity(n)

plt.show()

#Save figure
saving="C:/Users/maila/OneDrive/デスクトップ/保存先/LG1.png"
#plt.savefig(saving)