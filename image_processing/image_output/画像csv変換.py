from PIL import Image
import numpy as np
from matplotlib import pylab as plt
from scipy import ndimage
from matplotlib.colors import LinearSegmentedColormap


path="input_file/1/90v.png"
input=Image.open(path).convert('L')
input_resize=input.resize((500,500))
img = np.array(input) #画像のndarray化
print(np.size(img[1]))
print(img)
np.savetxt("C:/Users/maila/OneDrive/デスクトップ/program_file/MATLAB/ModePlot/3d_polarization2/1/90.csv", img,)