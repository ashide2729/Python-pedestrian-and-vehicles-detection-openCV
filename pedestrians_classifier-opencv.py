import cv2 
import imutils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
image = cv2.imread('pedestrians.jpg') 

image = imutils.resize(image, width=min(500, image.shape[1]))   

(points, _) = hog.detectMultiScale(image, winStride=(5, 5), padding=(5, 5), scale=1.02) 
   
for (x, y, w, h) in points: 
    cv2.rectangle(image, (x, y), (x+w, y+h),  (0, 255, 0), 2) 
  
cv2.imshow("Image", image) 
cv2.waitKey(0) 
cv2.imwrite('output.png',image)
cv2.destroyAllWindows() 

