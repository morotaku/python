import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#ファイルパス
fnum='27 5_0001'
name=fnum+'.ascii.csv'
path="C:/Users/maila/OneDrive/デスクトップ/研究室/田村さん＿スキルミオン/I135q_0001.ascii.csv"#+name

#csv読み込み
df=pd.read_csv(path,header=None)

#NaN除去
a=df.dropna(axis='columns')

#配列化
arr=np.array(a)
arr_max=np.amax(arr)
ar=arr/arr_max #規格化


#列、行の数
col=arr.shape[1]
lin=arr.shape[0]


X=0
Y=0
ar_sum=0
for i in range(col):
    for j in range(lin):
        X=X+i*ar[j][i] #x方向のモーメント
        Y=Y+j*ar[j][i] #y方向のモーメント
        ar_sum+=arr[j][i] #強度合計


#重心
X_g=int(X/ar_sum)
Y_g=int(Y/ar_sum)
#中心
X_c=int(col/2)
Y_c=int(lin/2)
#print('Xg=',X_g,"","Yg=",Y_g)
#切り取り範囲
L=75
X=0 #平行移動(左と上)
Y=0
#重心を中心に切り取り
#array_cut=ar[Y_g-L-Y:Y_g+L-Y,X_g-L-X:X_g+L-X]
print(Y_c-L-Y)
#画像中心を中心に切り取り
array_cut=arr[Y_c-L-Y:Y_c+L-Y,X_c-L-X:X_c+L-X]

#画像の設定
rb=LinearSegmentedColormap.from_list('name',['black','red'],N=5000) #カラースケール,N(色の段階)#DB631F(色)
fig = plt.figure(figsize=(3,3))
ax = fig.add_subplot(111)
ax.set_axis_off()

#画像表示
#plt.imshow(ar,interpolation = "none",cmap='gray',vmax=1,vmin=0) #元の画像
plt.imshow(array_cut,interpolation = "none",cmap='gray')#,vmax=0.89,vmin=0.08) #切取後
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

#保存
#saving="C:/Users/maila/OneDrive/デスクトップ/保存先/田村さん/1.png"#+fnum+".png" #保存先と保存名指定
#plt.savefig(saving,dpi=600) #画像保存
plt.show()