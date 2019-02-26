import cv2 as cv
import os
import numpy as np
import sys

def computeOpticalFlow(src_location,dest_location_opt,dest_location_img,store_name):
	print (src_location)
	frames = os.listdir(src_location)
	num = 0
	print (dest_location_opt)
	print (dest_location_img)
	try:
		frame1 = cv.imread(src_location + '\\'+ frames[num])

		print ("read", src_location +  frames[num])
		num += 1 
		prvs = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
		cv.imwrite(dest_location_img + frames[num]+ '.png',prvs)
	except:
		print ("Error")
		print (src_location)
	hsv = np.zeros_like(frame1)
	hsv[...,1] = 255
	while(1):
		try:	
			frame2 = cv.imread(src_location + '\\'+frames[num])
		
			num+=1
			next = cv.cvtColor(frame2,cv.COLOR_BGR2GRAY)
		except:
			print ("Over")
			break
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

		print (dest_location_img + str(store_name) + frames[num])
		print (dest_location_opt + str(store_name) + frames[num])

		# store_name = str(src_location)

		# store = ""
		# while store_name:
		# 	if store_name[-1] == '\':
		# 		break
		# 	else
		# 		store = store_name[-1] + store
		# 	store_name = store_name[:-1]

		cv.imwrite(dest_location_img + str(store_name) + frames[num]+ '.png',frame2)
		cv.imwrite(dest_location_opt + str(store_name) + frames[num]+ '.png',bgr)
		prvs = next
	
	cv.destroyAllWindows()


def baba(src_location,dest_location_opt,dest_location_img):
	main_dir = os.listdir(src_location)

	for i in main_dir:
		if i == 'A' or i == 'B' or i == 'preprocess.py':
			continue
		print (i)
		computeOpticalFlow(os.path.join(os.getcwd() , i), dest_location_opt, dest_location_img,i)
		

if __name__ == "__main__":
	#train_Set = "F:\\IIT D Competitoin\\UCSD_Anomaly_Dataset.tar\\UCSD_Anomaly_Dataset.v1p2\\UCSDped1\\Train\\Train"
	#train_Set_loc, train_set_dest_opt, train_set_dest_img = input(), input(), input() 
	train_Set_loc = os.getcwd()
	train_set_dest_opt = os.getcwd() + '\\B\\train\\'
	train_set_dest_img = os.getcwd() + '\\A\\train\\'
	#print (train_Set)
	train_Data_location_list = []

	# for i in range(1,31):
	# 	number = str(i)
	# 	while len(number) < 3:
	# 		number = '0' + number 
	# 	train_Data_location_list.append(train_Set+number)

	# print ("Train list obtained")
	
	train_file = os.getcwd()[:95]
	print(train_file)
	baba(train_Set_loc, train_set_dest_opt, train_set_dest_img)
