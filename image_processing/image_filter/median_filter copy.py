import numpy as np
import cv2
import sys

def main():

    print("OpenCV Version: " + str(cv2.__version__) + "\n")     

    #Load image data

    file="5_1.jpeg"
    image=cv2.imread(file,cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    if image is None:
        print("Cannot find image data : " + file)
        sys.exit()

    denoised_image1=medianFilter(image,5)
    cv2.imwrite('denoised_image1.jpg',denoised_image1)

def medianFilter(img,k):
  print(img.shape)
  w,h,c = img.shape
  size = k // 2

  # ０パディング処理
  _img = np.zeros((w+2*size,h+2*size,c), dtype=np.float64)
  _img[size:size+w,size:size+h] = img.copy().astype(np.float64)
  dst = _img.copy()

  # フィルタリング処理
  for x in range(w):
    for y in range(h):
      for z in range(c):
        dst[x+size,y+size,z] = np.median(_img[x:x+k,y:y+k,z])

  dst = dst[size:size+w,size:size+h].astype(np.uint8)

  return dst

if __name__ == "__main__":
    main()