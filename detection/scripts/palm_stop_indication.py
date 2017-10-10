
# Open Palm Stop Control Utility

import cv2

def run():
	
	# Modify classifier path
	open_palm = cv2.CascadeClassifier(r"C:\Users\abi.kuganesan\autonomous_applications\detection\classifiers\open_palm_classifier.xml")
	
	# Stream from external webcam 
	livestream = cv2.VideoCapture(0)	

	while (True):
		success, img = livestream.read()
		img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		bounding_boxes = open_palm.detectMultiScale(img, 1.3, 5)
		for (x,y,w,h) in bounding_boxes:
			cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
			
		cv2.imshow("Control Results", img)
		cv2.waitKey(10)
		
if __name__ == '__main__':
	run()
	
	