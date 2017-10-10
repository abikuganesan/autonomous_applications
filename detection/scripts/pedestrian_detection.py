
# HOG Pedestrian Detection Utility 

import cv2

def run():
	# External webcam source
	livestream = cv2.VideoCapture(1)

	hog = cv2.HOGDescriptor()
	hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

	while (True):
		success, img = livestream.read()
		row_size, col_size = img.shape[:2]
			
		(rect, weight) = hog.detectMultiScale(img, winStride=(16,16), padding=(8,8), scale=1.05)
		
		for (r, w) in zip(rect, weight):
			if w > 1:
				cv2.rectangle(img, (r[0], r[1]), (r[0] + r[2], r[1]+r[3]), (0, 255, 0), 2)
		cv2.imshow('Pedestrian Detection', img)
		cv2.waitKey(10)
		
if __name__ == '__main__':
	run()
	