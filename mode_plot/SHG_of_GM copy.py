import numpy as np
from math import *
from scipy.special import *
import matplotlib.pyplot as plt
import matplotlib as mlp
from matplotlib.colors import LinearSegmentedColormap

def main():

    lam=0.64 #%Wave length
    k=2*np.pi/lam #%Wave number
    #%%% Cavity parameter
    P=1
    Q=3
    R_cav=150*10**3
    L_cav=R_cav*np.sin(P*np.pi/Q)**2
    n0=10
    s0=floor(2*L_cav/lam)#%117188;
    M=5

    zr=np.sqrt(L_cav*(R_cav-L_cav)) #%Rayleigh length
    w=np.sqrt(lam*zr/pi)
    z=0*zr #;%*1.75;%input('position: '); %Beam position
    R=z+zr**2/z #%Beam curvature
    W=w*np.sqrt((1+(z/zr)**2)) #%Beam size
  
    #x-y　coordinate
    N=1000
    L=floor(W)*10 #Display range
    X=np.linspace(-L,L,N)
    Y=np.linspace(-L,L,N)
    x,y=np.meshgrid(X,Y)
    r=np.sqrt(x**2+y**2)
    phi1=np.arctan(y,x)

    #%%% SHG parameter
    z1=0 #%crystal surface (μm)
    z2=1000 #% crystal end-surface (μm)
    A=1
    K=0.1 #%efficiency of SHG in the unit length
    print(SHG_LG(5,0,z,z1,z2,zr,A,K,w,r,phi1,lam))
    
def SHG_LG_degenerate(x,y,z,zr,n0,M,r,phi,w,lam,A,Q,z1,z2):
    k=2*np.pi/lam
    Z=(x**2+y**2)*z/(2*(z**2+zr**2))
    phi0=0
    E_lg=0
    for K in range(0,M+1,1):
        E_lg=E_lg+np.sqrt(factorial(M)/(factorial(K)*factorial(M-K)))*np.exp(1j*K*phi0)*SHG_LG(n0+Q*K,0,z,z1,z2,zr,A,K,w,r,phi,lam)
        

    E=1/2**(M/2)*np.exp(-1j*k*Z)*E_lg
    y=A**2*K*zr/(2*np.sqrt(2)*w)*(2*1+1j*log((zr**2+z2**2)/(zr**2+z1**2)))**E
    return y
    
def SHG_LG(n,m,z,z1,z2,zr,A,K,w,r,phi,lam):
    E_lg=0
    for i in range(0,n+1,1):
        for j in range(0,m+1,1):
            #E_lg=E_lg+(c_nm(n,m).^2)/(c_nm(2*i,0)*c_nm(0,2*j))*e_nm(n,i)*e_nm(m,j).*LGmode(2*min(i,j),2*i-2*j,r,phi,z,w,lam);
            E_lg=E_lg+LGmode(2*i-2*j,2*min([i,j]),r,phi,z,w,lam)#+(c_nm(n,m)**2)/(c_nm(2*i,0)*c_nm(0,2*j))*e_nm(n,i)*e_nm(m,j)*LGmode(2*i-2*j,2*min([i,j]),r,phi,z,w,lam)

    e_lg=E_lg/np.amax(abs(E_lg))
    E=e_lg
    return E#./max(max(abs(E)));
    
def LGmode(l,p,r,phi,z,w,lam):
    k=2*np.pi/lam 
    zr=w**2*k/2
    R=z+zr**2/z
    W=w*np.sqrt(1+(z/zr)**2)
    gouy=1j*(2*p+abs(l)+1)*np.arctan2(z,zr)
    LG1=d_lp(l,p)/W*(np.sqrt(2)*r/W)**abs(l)*(-1)**p*np.exp(-1j*l*phi)*laguerre(p,abs(l),2*r**2/(W**2))*np.exp(gouy)*np.exp(-r**2*((1/W**2)+(1j*k/(2*R))))
    LG=abs(LG1**2)
    #y=LG/np.amax(LG)
    return LG1

def laguerre(p,l,x):
    Lag=0
    for k in range(0,p+1):
        Lag += (-1)**(k+abs(l))*((factorial(p+abs(l))**2*x**k)/(factorial(k)*factorial(k+abs(l))*factorial(p-k)))
    return Lag

def c_nm(n,m):
    return np.sqrt(2/(np.pi*factorial(n)*factorial(m)))*2**(-(n+m)/2)

def d_lp(l,p):
    return np.sqrt(2*factorial(p)/(np.pi*factorial(abs(l)+p)))

def e_nm(n,m):
    e=0
    for a in range(0,floor(n/2)+1,1):
        for b in range(0,floor(n/2)+1,1):
            for c in range(0,m+1,1):
                #%e1=2**(a+b-n-2*m)*(-1)**(a+b+c)*factorial(n)**2*Product(2*n+2*m-2*a-2*b-2*c,n+m-a-b-c)
                e1=2**(a+b-n-2*m)*(-1)**(a+b+c)*factorial(n)**2*factorial(2*n+2*m-2*a-2*b-2*c)/factorial(n+m-a-b-c)
                e2=factorial(n-2*a)*factorial(n-2*b)*factorial(2*m-2*c)*factorial(a)*factorial(b)*factorial(c)
                e=e1/e2+e
    return e

def gouy_phase(z,zr,n,m):
    return np.exp(1j*(m+n+1)*np.arctan(z/zr))

def Product(a,b):
    P=1
    if a>b:
        for i in range(b+1,a+1,1):
            P=P*i
    elif a<b:
        for i in range(a+1,b+1,1):
            P=P/i
    elif a==b:
        P=1
    return P

if __name__ == "__main__":
    main()