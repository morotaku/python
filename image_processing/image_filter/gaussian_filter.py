import numpy as np
import cv2
import sys

def main():
        
    print("OpenCV Version: " + str(cv2.__version__) + "\n")     

    #Load image data
    
    file="5_1.jpeg"
    image=cv2.imread(file,cv2.IMREAD_COLOR)

    if image is None:
        print("Cannot find image data : " + file)
        sys.exit()
    
    denoised_image1=cv2.GaussianBlur(image,ksize=(3, 3), sigmaX=0.85)
    cv2.imwrite('gaussian_image1.jpg',denoised_image1)

if __name__ == "__main__":
    main()