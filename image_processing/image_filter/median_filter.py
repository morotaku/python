import numpy as np
import cv2
import sys
from matplotlib import pylab as plt

def main():

    print("OpenCV Version: " + str(cv2.__version__) + "\n")     

    #Load image data

    file="5_1.jpeg"
    image=cv2.imread(file,cv2.IMREAD_COLOR)
    print(image)
    if image is None:
        print("Cannot find image data : " + file)
        sys.exit()

    denoised_image1=cv2.medianBlur(image,5)
    cv2.imwrite('denoised_image1.jpg',denoised_image1)
    plt.imshow(image,interpolation = "none",cmap='hot')#,vmax=1,vmin=0.01)
    plt.show()
if __name__ == "__main__":
    main()