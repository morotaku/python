import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import sys
import colorsys

def main():
    L=400#ファイルパス
    ar1=input_array("input_file/26 2_0001.ascii.csv")#+name
    ar2=input_array("input_file/26 4_0001.ascii.csv")#+name
    ar3=input_array("input_file/26 3_0001.ascii.csv")
    ar4=input_array("input_file/26 5_0001.ascii.csv")
    
    ###重心
    X_g1,Y_g1=center_of_gravity(ar1)
    X_g2,Y_g2=center_of_gravity(ar2)
    X_g3,Y_g3=center_of_gravity(ar3)
    X_g4,Y_g4=center_of_gravity(ar4)

    # ###オリジナルイメージ
    # fig1 = plt.figure(figsize=(10,6))
    # or1=fig1.add_subplot(221)
    # image_show(ar1,np.amax(ar1),np.amin(ar1))
    # or1.scatter(X_g1, Y_g1, c='cyan', s=20)

    # or2=fig1.add_subplot(222)
    # image_show(ar2,np.amax(ar2),np.amin(ar2))
    # or2.scatter(X_g2, Y_g2, c='cyan', s=20)

    # or3=fig1.add_subplot(223)
    # image_show(ar3,np.amax(ar3),np.amin(ar3))
    # or3.scatter(X_g3, Y_g3, c='cyan', s=20)

    # or4=fig1.add_subplot(224)
    # image_show(ar4,np.amax(ar4),np.amin(ar4))
    # or4.scatter(X_g4, Y_g4, c='cyan', s=20)


    #Cut image
    array_cut1=image_cut(ar1,L,0,50)
    array_cut2=image_cut(ar2,L,0,50)
    array_cut3=image_cut(ar3,L,0,0)
    array_cut4=image_cut(ar4,L,0,0)

    #Cut image show
    fig2 = plt.figure(figsize=(10,6))
    cut1=fig2.add_subplot(221)
    image_show(array_cut1,np.amax(array_cut1),np.amin(array_cut1))
    cut2=fig2.add_subplot(222)
    image_show(array_cut2,np.amax(array_cut2),np.amin(array_cut2))
    cut3=fig2.add_subplot(223)
    image_show(array_cut3,np.amax(array_cut3),np.amin(array_cut3))
    cut4=fig2.add_subplot(224)
    image_show(array_cut4,np.amax(array_cut4),np.amin(array_cut4))

    #Denoice image
    denoised_image1=medianFilter(array_cut1,3)
    denoised_image2=medianFilter(array_cut2,3)
    denoised_image3=medianFilter(array_cut3,3)
    denoised_image4=medianFilter(array_cut4,3)

    denoised_image1_n=denoised_image1/np.amax(denoised_image1)
    denoised_image2_n=denoised_image2/np.amax(denoised_image2)
    denoised_image3_n=denoised_image3/np.amax(denoised_image3)
    denoised_image4_n=denoised_image4/np.amax(denoised_image4)
    
    #Denoice image show
    fig3 = plt.figure(figsize=(10,6))
    dn1=fig3.add_subplot(221)
    image_show(denoised_image1_n,1,0)
    dn1.set_title('I00')
    dn2=fig3.add_subplot(222)
    image_show(denoised_image2_n,1,0)
    dn2.set_title('I90')
    dn3=fig3.add_subplot(223)
    image_show(denoised_image3_n,1,0) 
    dn3.set_title('I45')  
    dn4=fig3.add_subplot(224)
    image_show(denoised_image4_n,1,0)
    dn4.set_title('I135')

    # ### Stokes parameter
    # I00 = denoised_image1
    # I90 = denoised_image2
    # I45 = denoised_image3
    # I135 = denoised_image4
    # mx=np.amax(I00+I90)
    # # I00=I00/mx
    # # I90=I90/mx
    # # I45=I45/mx
    # # I135=I135/mx
    
    
    # s0=(I00+I90)
    # s1=(I00-I90)
    # s2=(I45-I135)
    # s3=np.sqrt(abs(s0**2-s1**2-s2**2))
    # # s0=(I00+I90)/np.amax(s0)
    # # s1=(I00-I90)/np.amax(s1)
    # # s2=(I45-I135)/np.amax(s2)
    # # s3=s0-s1-s2/np.amax(s3)
    # # if s1 >= 0:
    # #    s3=np.sqrt(abs(s0**2-s1**2-s2**2))
    # # else:
    # #    s3=-np.sqrt(abs(s0**2-s1**2-s2**2))
    

    
    # s0_n=s0/np.amax(s0+0.0000001)
    # s1_n=s1/np.amax(s0+0.0000001)
    # s2_n=s2/np.amax(s0+0.0000001)
    # s3_n=s3/np.amax(s0+0.0000001)
    

    # fig4 = plt.figure(figsize=(10,6))
    # # st1=fig4.add_subplot(221)
    # im1=image_show2(s0_n)#,np.amax(s0),-np.amax(s0))
    # #fig4.set_title('s0')
    # plt.colorbar()
    # plt.clim(-np.amax(s0_n),np.amax(s0_n))

    # fig5 = plt.figure(figsize=(10,6))
    # st2=fig5.add_subplot(131)
    # im2=image_show2(s1_n)#,np.amax(s1),-np.amax(s1))
    # st2.set_title('s1')
    # plt.colorbar()
    # plt.clim(-np.amax(s0_n),np.amax(s0_n))

    # st3=fig5.add_subplot(132)
    # im3=image_show2(s2_n)
    # st3.set_title('s2')
    # plt.colorbar()
    # plt.clim(-np.amax(s0_n),np.amax(s0_n))

    # st4=fig5.add_subplot(133)
    # im4=image_show2(s3_n)#,np.amax(s0),-np.amax(s0))
    # st4.set_title('s3')
    # plt.colorbar()
    # plt.clim(-np.amax(s0_n),np.amax(s0_n))
    #plt.clim(np.amin(s0),np.amax(s1_n))

    plt.show() 
    #plt.savefig(saving,dpi=600) #画像保存
    
    

    #画像表示
    #plt.imshow(ar,interpolation = "none",cmap='gray',vmax=1,vmin=0) #元の画像

    #保存
    np.savetxt("保存先/0.csv", denoised_image1, delimiter=",")
    np.savetxt("保存先/90.csv", denoised_image2, delimiter=",")
    # np.savetxt("保存先/45.csv", denoised_image3, delimiter=",")
    # np.savetxt("保存先/135.csv", denoised_image4, delimiter=",")
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
