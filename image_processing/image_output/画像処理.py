from PIL import Image
import numpy as np
from matplotlib import pylab as plt
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap
def main():
    L=300
    path="input_file/19 2_0001.jpeg"
    input=Image.open(path).convert('L')
    input_resize=input.resize((1000,1000))
    img = np.array(input_resize) #画像のndarray化
    img_n=img/(np.amax(img)) #強度規格化

    #Show original image
    original_image(img_n)
    #Cut image
    array_cut=image_cut(img_n,L)
    array_cut=array_cut/np.amax(array_cut)
    cut_image(array_cut,1,0)
    #Denoice
    denoised_image=medianFilter(array_cut,3)
    cut_image(denoised_image,0.9,0.2)

    
    plt.show()
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
    binary_arr = np.where(array > threshold, 0.5, 0)
    return binary_arr


#重心
def center_of_gravity(arr):
        array=binarize(arr,0.25)
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
def image_cut(array,L):
    #切り取り範囲
    X=0 #平行移動(左と上)
    Y=0
    X_o,Y_o=center_of_gravity(array)
    array_cut=array[Y_o-L+Y:Y_o+L+Y,X_o-L+X:X_o+L+X]
    return array_cut

def original_image(array):
    rb=LinearSegmentedColormap.from_list('name',['black','red'],N=20000) #カラースケール,N(色の段階)#DB631F(色)
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    X_o,Y_o=center_of_gravity(array)
    print(X_o,Y_o)
    plt.imshow(array,interpolation = "none",cmap='hot',vmax=1,vmin=0.01) #切取後
    ax.scatter(X_o, Y_o, c='cyan', s=20)
    #plt.show()

def cut_image(array,vmax,vmin):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    plt.imshow(array,interpolation = "none",cmap='hot',vmax=vmax,vmin=vmin) #切取後
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    #保存

if __name__=="__main__":
    main()