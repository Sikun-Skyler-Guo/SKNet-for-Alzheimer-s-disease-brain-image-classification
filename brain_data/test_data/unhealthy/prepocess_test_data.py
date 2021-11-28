import cv2
import numpy as np

for i in range(1, 2):
    index = str(i)
    file_name = index + ".png"
    image = cv2.imread(filename=file_name)
    print("shape:", image.shape)
    print("shape:", type(image))
    print(image)
cv2.imshow('image', image)
cv2.waitKey(0)