import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import sys
import colorsys

def main():
    n=4
    L=300    #ファイルパス
    ar1=input_array("input_file/26 2_0001.ascii.csv")#+name

    
    ###重心
    X_g1,Y_g1=center_of_gravity(ar1)


    ###オリジナルイメージ
    fig1 = plt.figure(figsize=(10,6))
    or1=fig1.add_subplot(111)
    image_show(ar1,np.amax(ar1),np.amin(ar1))
    or1.scatter(X_g1, Y_g1, c='cyan', s=20)
    or1.invert_yaxis()



    #Cut image
    array_cut1=image_cut(ar1,L,0,0)


    #Cut image show
    fig2 = plt.figure(figsize=(10,6))
    cut1=fig2.add_subplot(111)
    image_show(array_cut1,np.amax(array_cut1),np.amin(array_cut1))


    #Denoice image
    denoised_image1=medianFilter(array_cut1,3)

    
    #Denoice image show
    fig3 = plt.figure(figsize=(10,6))
    dn1=fig3.add_subplot(111)
    image_show(denoised_image1,np.amax(denoised_image1),np.amin(denoised_image1))

    denoised_n=denoised_image1/np.amax(denoised_image1)
    denoised_binary=binarize(denoised_n, 0.75)
    row_indices, col_indices = np.where(denoised_binary == 1)
    print(np.amin(col_indices),np.amax(col_indices))
    slope, intercept=np.polyfit(col_indices, row_indices, 1)
    x_f = np.linspace(0, len(denoised_image1[0][:]), len(denoised_image1[0][:]))
    plt.plot(x_f,np.floor(x_f*slope+intercept))
    x_test=()
    y=np.floor(x_f*slope+intercept)
    plt.plot(x_f,y)
    print(slope, intercept)

    point_x=[0]*(n+1)
    point_y=[0]*(n+1)
    point_0=200
    point_n=400
    for i in range(n+1):
        #interval=(np.amax(col_indices)-np.amin(col_indices))/n
        interval=(point_n-point_0)/n
        point_x[i]=point_0+i*interval
        point_y[i]=point_x[i]*slope+intercept
 
    boundary_x=[0]*n
    boundary_y=[0]*n
    for i in range(n):
        boundary_x[i],boundary_y[i]=(point_x[i]+point_x[i+1])/2, (point_y[i]+point_y[i+1])/2

    slope_inv=-1/slope
    bound=[0]*n
    for i in range(n):
        bound[i]=slope_inv*x_f-slope_inv*boundary_x[i]+boundary_y[i]

    plt.scatter(boundary_x,boundary_y)
    print(boundary_x,boundary_y)

    for i in range(n):
        plt.plot(x_f,bound[i])

    dn1.invert_yaxis()

    fig4 = plt.figure(figsize=(5,5))
    sc1=fig4.add_subplot(111)
    phase=np.zeros((2*L,2*L))
    for x in range(2*L):
        for y in range(2*L):
            if y>slope_inv*x-slope_inv*boundary_x[0]+boundary_y[0]:
                phase[y][x]=1j*np.pi
            elif slope_inv*x-slope_inv*boundary_x[1]+boundary_y[1]>y>slope_inv*x-slope_inv*boundary_x[2]+boundary_y[2]:
                phase[y][x]=1j*np.pi
            elif slope_inv*x-slope_inv*boundary_x[3]+boundary_y[3]>y:
                phase[y][x]=1j*np.pi
    plt.imshow(np.angle(phase))
    sc1.invert_yaxis()
    # fig4 = plt.figure(figsize=(5,5))
    # sc1=fig4.add_subplot(111)
    # image_show(denoised_binary,np.amax(denoised_binary),np.amin(denoised_binary))
    # plt.scatter(col_indices, row_indices)
    # sc1.set_aspect("equal")

    
    
    plt.show() 
    # dn1.set_title('I00')
    # dn2=fig3.add_subplot(222)
    # image_show(denoised_image2,np.amax(denoised_image2),np.amin(denoised_image2))
    # dn2.set_title('I90')
    # dn3=fig3.add_subplot(223)
    # image_show(denoised_image3,np.amax(denoised_image3),np.amin(denoised_image3)) 
    # dn3.set_title('I45')  
    # dn4=fig3.add_subplot(224)
    # image_show(denoised_image4,np.amax(denoised_image4),np.amin(denoised_image4))
    # dn4.set_title('I135')



    #画像表示
    #plt.imshow(ar,interpolation = "none",cmap='gray',vmax=1,vmin=0) #元の画像

    #保存
    # np.savetxt("保存先/1.csv", denoised_image1, delimiter=",")
    #saving="C:/Users/maila/OneDrive/デスクトップ/保存先/田村さん/1.png"#+fnum+".png" #保存先と保存名指定
    #plt.savefig(saving,dpi=600) #画像保存


####################################################################################################
def input_array(path):
    #csv読み込みdayo
    df=pd.read_csv(path,header=None)
    #NaN除去dayo
    a=df.dropna(axis='columns')
    #配列化
    arr=np.array(a,dtype='uint64')
    #arr=arr[0:1199,700:1599]
    arr_max=np.amax(arr)
    ar=arr/arr_max #規格化
    return arr

#メディアンフィルター
def medianFilter(img,k):
    w,h = img.shape
    size = k // 2

    # ０パディング処理
    _img = np.zeros((w+2*size,h+2*size), dtype=np.float64)
    _img[size:size+w,size:size+h] = img.copy().astype(np.float64)
    dst = _img.copy()

    # フィルタリング処理
    for x in range(w):
        for y in range(h):
            dst[x+size,y+size] = np.median(_img[x:x+k,y:y+k])
    
    dst = dst[size:size+w,size:size+h].astype(np.float64)

    return dst

#2値化
def binarize(array, threshold):
    # 閾値を超える要素を1に、それ以外を0にする
    arr=array/np.amax(array)
    binary_arr = np.where(arr > threshold, 1, 0)
    return binary_arr


#重心
def center_of_gravity(arr):
        array=binarize(arr,0.5)
        #array=binarize(arr,np.amax(arr)/2)
        col=array.shape[1]
        lin=array.shape[0]
        X=0
        Y=0
        ar_sum=0
        for i in range(col):
            for j in range(lin):
                X=X+i*array[j][i] #x方向のモーメント
                Y=Y+j*array[j][i] #y方向のモーメント
                ar_sum+=array[j][i] #強度合計

        X_g=int(X/ar_sum)
        Y_g=int(Y/ar_sum)
        return X_g,Y_g
    
#画像の中心
def center_of_image(array):
    col=array.shape[1]
    lin=array.shape[0]
    X_c=int(col/2)
    Y_c=int(lin/2)
    print(X_c,Y_c)
    return X_c,Y_c

#画像切り取り
def image_cut(array,L,X,Y):
    #切り取り範囲
    X_o,Y_o=center_of_gravity(array)
    array_cut=array[Y_o-L+Y:Y_o+L+Y,X_o-L+X:X_o+L+X]
    return array_cut


def image_show(array,vmax,vmin):
    plt.imshow(array,interpolation = "none",cmap='jet',vmax=vmax,vmin=vmin) #切取後
    #plt.show()

def image_show2(array):
    plt.imshow(array,interpolation = "none",cmap='jet')#,vmax=vmax,vmin=vmin) #切取後
    #plt.show()

if __name__ == "__main__":
    main()

#おいしさが止まらない
#Latest Update 2023/11/23 Author Takuya Morohasi
