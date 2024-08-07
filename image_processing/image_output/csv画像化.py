import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import sys
import colorsys

def main():
    L=300    #ファイルパス
    fnum="5 3"
    path="input_file/4/0.csv"#+name
    saving="C:/Users/maila/OneDrive/デスクトップ/保存先/a.png" #保存先と保存名指定

    #csv読み込み
    df=pd.read_csv(path,header=None)
    #NaN除去
    a=df.dropna(axis='columns')
    
    
    #配列化
    arr=np.array(a,dtype='uint64')
    #arr=arr[0:1199,700:1599]
    arr_max=np.amax(arr)
    ar=arr/arr_max #規格化

    #for i in range(ar.shape[0]-1):
    #    for j in range(ar.shape[1]-1):
    #        if ar[i,j]>0.1:
    #            ar[i,j]=0.001

    #Show original image
    original_image(ar)
    
    plt.savefig('input_file/1/0.png',bbox_inches="tight") #画像保存
    #Cut image
    array_cut=image_cut(ar,L,300,0)
    cut_image(array_cut,1,0)
    #Denoice
    denoised_image=medianFilter(array_cut,5)
    cut_image(denoised_image,0.8,0)
    
    
    #plt.savefig(saving,dpi=600) #画像保存
    plt.show()

    #画像表示
    #plt.imshow(ar,interpolation = "none",cmap='gray',vmax=1,vmin=0) #元の画像

    #保存
    #saving="C:/Users/maila/OneDrive/デスクトップ/保存先/田村さん/1.png"#+fnum+".png" #保存先と保存名指定
    #plt.savefig(saving,dpi=600) #画像保存


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
    binary_arr = np.where(array > threshold, 1, 0)
    return binary_arr


#重心
def center_of_gravity(arr):
        array=binarize(arr,0.5)
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
    return X_c,Y_c

#画像切り取り
def image_cut(array,L,X,Y):
    #切り取り範囲
    X_o,Y_o=center_of_gravity(array)
    array_cut=array[Y_o-L+Y:Y_o+L+Y,X_o-L+X:X_o+L+X]
    return array_cut

def original_image(array):
    rb=LinearSegmentedColormap.from_list('name',['black','red'],N=20000) #カラースケール,N(色の段階)#DB631F(色)
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    X_o,Y_o=center_of_gravity(array)
    print(X_o,Y_o)
    plt.imshow(array,interpolation = "none",cmap='gray',vmax=1,vmin=0) #切取後
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #ax.scatter(X_o, Y_o, c='cyan', s=20)
    
    #plt.show()

def cut_image(array,vmax,vmin):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    plt.imshow(array,interpolation = "none",cmap='hot',vmax=vmax,vmin=vmin) #切取後
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #保存


if __name__ == "__main__":
    main()
