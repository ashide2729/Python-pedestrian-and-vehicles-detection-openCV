import cv2 
import imutils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 

cars_cascade = cv2.CascadeClassifier('cars.xml')

image = cv2.imread('pedestrians.jpg') 

image = imutils.resize(image, width=min(500, image.shape[1]))   

(points, _) = hog.detectMultiScale(image, winStride=(5, 5), padding=(5, 5), scale=1.02) 
   
for (x, y, w, h) in points: 
    cv2.rectangle(image, (x, y), (x+w, y+h),  (0, 255, 0), 2)

cars = cars_cascade.detectMultiScale(image, 1.015, 5)

for (x, y, w, h) in cars:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    roi_color = image[y:y+h, x:x+w] 
  
cv2.imshow("Image", image) 
cv2.waitKey(0) 
cv2.imwrite('output.png',image)
cv2.destroyAllWindows() 

