import cv2 as cv
import os
import numpy as np
import sys





def computeOpticalFlow(src_location,dest_location):
	print (src_location)
	frames = os.listdir(src_location)
	num = 0
	frame1 = cv.imread(src_location + '\\'+ frames[num])
	print ("read", src_location +  frames[num])
	num += 1 
	prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255
	while(1):
		try:	
			frame2 = cv.imread(src_location + '\\'+frames[num])
		except:
			print ("Over")
			break
		num+=1
		next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
		flow = cv.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
		mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
		hsv[...,0] = ang*180/np.pi/2
		hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
		bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
		cv.imshow('orig_image',frame2)
		cv.imshow('frame2',bgr)
		k = cv.waitKey(30) & 0xff
		if k == 27:
		    break
	    
	    kap = str(num)
	    while(len(kap)) < 3:
	    	kap = '0' + kap

		cv.imwrite( kap + '.png',bgr)
		prvs = next
	
	cv.destroyAllWindows()


if __name__ == "__main__":
	train_Set = "F:\\IIT D Competitoin\\UCSD_Anomaly_Dataset.tar\\UCSD_Anomaly_Dataset.v1p2\\UCSDped1\\Train\\Train"
	print (train_Set)
	train_Data_location_list = []

	# for i in range(1,31):
	# 	number = str(i)
	# 	while len(number) < 3:
	# 		number = '0' + number 
	# 	train_Data_location_list.append(train_Set+number)

	# print ("Train list obtained")
	
	train_file = os.getcwd()[:95]
	print(train_file)
	computeOpticalFlow(train_file,"")
