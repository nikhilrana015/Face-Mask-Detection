# Importing the Necessary Libraries. 
import cv2
import numpy as np 
import imutils
import os
import time
from keras.applications.mobilenet_v2 import preprocess_input as mobile_pre
from keras.applications.vgg16 import preprocess_input as vgg16_pre
from keras.applications.xception import preprocess_input as xcep_pre
from keras.models import load_model
from keras.preprocessing.image import img_to_array


# Importing the Face- Detector Models
path_to_protxt = os.path.join(os.getcwd(),'face_detector','deploy.prototxt')
path_to_caffe_model = os.path.join(os.getcwd(),'face_detector','resnet_caffe.caffemodel')

# Path to Load Different Model.
path_load_model_1 = os.path.join(os.getcwd(), 'Save_Model','model1_mobileNet.h5')
path_load_model_2 = os.path.join(os.getcwd(), 'Save_Model','model2_vgg16.h5')
path_load_model_3 = os.path.join(os.getcwd(), 'Save_Model','model3_xception.h5')


net = cv2.dnn.readNetFromCaffe(path_to_protxt, path_to_caffe_model)

# Load the Different Models
# Model1: MobileNet_V2
# Model2: VGG16
# Model3: Xception

model1 = load_model(path_load_model_3)           
# model2 = load_model(path_load_model_2)
# model3 = load_model(path_load_model_3)



# VideoCapture for opening the camera.
cap = cv2.VideoCapture(0)
time.sleep(2.0)       

'''
This is basically adding to make video of the frames.
Fourcc code for the Video.
output.avi  ->> Video name + format
fourcc  ->> FourCC code
Frames per sec in the output video --> 20
frame width and height
cv2.VideoWriter_fourcc('M','J','P','G')

'''
# fourcc = cv2.VideoWriter_fourcc(*'MP4V')
# output = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))
writer = None


while(True):

	# Read the frame and resize it to max_width of 400.

	_, frame = cap.read()
	frame = imutils.resize(frame, width=400)

	(h,w) = frame.shape[:2]

	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
								(300, 300), (104.0, 177.0, 123.0))


	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()


	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence < 0.5:
			continue


		# compute the (x, y)-coordinates of the bounding box for the
		# object
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")


		# We get the box which contains the user's face.
		# We take our region of interest and change to RGB format
		# After resize to (224,224,3) to make them compatible related to model input.
		# Convert the PIL Image to Array

		face = frame[startY:endY,startX:endX]
		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize( face, (224,224))
		face = img_to_array(face)
		

		# assert face.shape == (224,224,3)
		# print(face.shape)

		# preprocess the face pixels according to the use of pretrained model.
		# Changing the dimension --> (1,224,224,3)
		face = mobile_pre(face)
		face = face[np.newaxis,:]

		# print(np.argmax(model.predict(face), axis=1)
		prediction = np.argmax(model1.predict(face), axis=1)
		y = startY - 10 if startY - 10 > 10 else startY + 10

		if prediction==0:

			text = "No Mask-Danger"
			color = (0, 0, 255)


		else:

			text = "Masked-Safe"
			color = (0, 255, 0)

		cv2.rectangle(frame, (startX,startY), (endX,endY), color, 2)
		cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)


	# output.write(frame)
	cv2.imshow("Frame", frame)

	if cv2.waitKey(25) & 0xFF == ord('q'):
		break;

	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter("output.avi", fourcc, 25, (frame.shape[1], frame.shape[0]), True)

	if writer is not None:
		writer.write(frame)


cap.release()
# output.release()
writer.release()
cv2.destroyAllWindows()


# https://youtu.be/anvCtnFunYM

		