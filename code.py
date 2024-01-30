import cv2
import numpy as np import os
from matplotlib import pyplot as plt import time
import mediapipe as mp
mp_holistic = mp.solutions.holistic	# holistic model-downloading the model and leveraging it to make detections
mp_drawing = mp.solutions.drawing_utils # drawing utilities-It makes easier to actually draw the key-points on our face.
# function to make detection
def mediapipe_detection(image, model):
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert color space from BGR to RGB
image.flags.writeable = False	# image is no longer writeable results = model.process(image)		# predictions image.flags.writeable = True	 # image is writeable
image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert color space from RGB to BGR
return image, results
def draw_styled_landmarks(image, results):


# face connections
mp_drawing.draw_landmarks(image,	results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,

 

circle_radius=1), circle_radius=1))
 
mp_drawing.DrawingSpec(color=(255, 255, 86), thickness=1,

mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1,
 
# pose connections
mp_drawing.draw_landmarks(image,	results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
mp_drawing.DrawingSpec(color=(255, 86, 170), thickness=2,
 
circle_radius=4), circle_radius=2))
 

mp_drawing.DrawingSpec(color=(86, 255, 255), thickness=2,
 
# left hand connections
mp_drawing.draw_landmarks(image,	results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
mp_drawing.DrawingSpec(color=(0,	127,	255),	thickness=2,
 
circle_radius=4), circle_radius=2))
 

mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
 
# right hand connections
mp_drawing.draw_landmarks(image,	results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
mp_drawing.DrawingSpec(color=(255,	0,	0),	thickness=2,
circle_radius=4),



 
mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
circle_radius=2))
cap = cv2.VideoCapture(0) # read the feed from webcam device


# set mediapipe model
with	mp_holistic.Holistic(min_detection_confidence	=	0.5, min_tracking_confidence = 0.5) as holistic:
while cap.isOpened():	# double-check for webcam access & loop through all frames

# read feed/frames from webcam ret, frame = cap.read()

# make detections
image, results = mediapipe_detection(frame, holistic)


# draw formatted landmarks draw_styled_landmarks(image, results)

# show image to screen cv2.imshow('OpenCV Feed', image)

# break gracefully if hit 'q' on keyboard if cv2.waitKey(10) & 0xFF == ord('q'):
break
cap.release()	# release webcam cv2.destroyAllWindows() # close down all frames

 
draw_styled_landmarks(frame, results) # apply landmarks plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # color conversion def extract_keypoints(results):

pose	=	np.array([[res.x,	res.y,	res.z,	res.visibility]	for	res	in results.pose_landmarks.landmark]).flatten() \
if results.pose_landmarks else np.zeros(33*4)


face	=	np.array([[res.x,	res.y,	res.z]	for	res	in results.face_landmarks.landmark]).flatten() \
if results.face_landmarks else np.zeros(468*3)


lh	=	np.array([[res.x,	res.y,	res.z]	for	res	in results.left_hand_landmarks.landmark]).flatten() \
if results.left_hand_landmarks else np.zeros(21*3)


rh	=	np.array([[res.x,	res.y,	res.z]	for	res	in results.right_hand_landmarks.landmark]).flatten() \
if results.right_hand_landmarks else np.zeros(21*3)


return np.concatenate([pose, face, lh, rh]) extract_keypoints(results) extract_keypoints(results).shape
data_path = os.path.join('MP_Data') #path for exported data







 
actions	=	np.array(['HELLO',		'THANKS',	'I	LOVE U','YES','NO','PLEASE','GOOD	BYE','SORRY','YOU			ARE
WELCOME','FAMILY','HOUSE','LOVE']) #actions to be detected
no_sequences = 10 #no. of videos collected for each action sequence_length = 10 #frame length of each video
for action in actions:
for sequence in range(no_sequences): try:
os.makedirs(os.path.join(data_path, action, str(sequence))) except:
pass
cap = cv2.VideoCapture(0) #to access webcam device

# set mediapipe model
with	mp_holistic.Holistic(min_detection_confidence	=	0.5, min_tracking_confidence = 0.5) as holistic:
#NEW LOOP
for action in actions: # loop through actions
for sequence in range(no_sequences): # loop through sequences/videos for frame_no in range(sequence_length): # loop through video length

# read frames
ret, frame = cap.read()

# make detections

 
image, results = mediapipe_detection(frame, holistic)

# draw landmarks draw_styled_landmarks(image, results)

# NEW collection wait logic if frame_no == 0:
cv2.putText(image, 'COLLECTING NOW...', (120, 200),
cv2.FONT_HERSHEY_PLAIN,	2,	(0,	255,	0),	3,
cv2.LINE_AA)
cv2.putText(image,  'Collecting  frames  for  Action:  {}  &  Video:
{}'.format(action, sequence), (15, 12),
cv2.FONT_HERSHEY_PLAIN,	1,	(0,	0,	255),	1,
cv2.LINE_AA)
cv2.imshow('OpenCV Feed', image) # show image on screen cv2.waitKey(1500) # 1.5 seconds break
else:
cv2.putText(image,  'Collecting  frames  for  Action:  {}  &  Video:
{}'.format(action, sequence), (15, 12),
cv2.FONT_HERSHEY_PLAIN,	1,	(0,	0,	255),	1,
cv2.LINE_AA)
cv2.imshow('OpenCV Feed', image) # show image on screen

# NEW extract key-points
keypoints = extract_keypoints(results)
npy_path = os.path.join(data_path, action, str(sequence), str(frame_no)) np.save(npy_path, keypoints)

 
# break gracefully if hit 'q' on keyboard if cv2.waitKey(10) & 0xFF == ord('q'):
break
cap.release()	# release webcam cv2.destroyAllWindows() # close down all frames
from sklearn.model_selection import train_test_split from tensorflow.keras.utils import to_categorical
label_map = {label:num for num, label in enumerate(actions)} label_map
sequences, labels = [], [] # blank arrays for action in actions:
for sequence in range(no_sequences): window = [] # blank array
for frame_num in range(sequence_length):
res	=	np.load(os.path.join(data_path,	action,	str(sequence), "{}.npy".format(frame_num)))
window.append(res) sequences.append(window) labels.append(label_map[action])
np.array(sequences).shape np.array(labels).shape
X = np.array(sequences)
y = to_categorical(labels).astype(int) # one-hot-encoding
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.05) from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

 
from tensorflow.keras.callbacks import TensorBoard log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir) model = Sequential()

# add 3 set of LSTM layers
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(10, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu')) model.add(LSTM(64, return_sequences=False, activation='relu'))

# add 3 Dense layers model.add(Dense(64, activation='relu')) model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # actions layer model.compile(optimizer='Adam',	loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=1000, callbacks=[tb_callback]) model.summary()
res = model.predict(X_test)#Now, we will make the predictions on the test data based on this model.
actions[np.argmax(res[2])]#We can check the prediction for a random action by unpacking the results
actions[np.argmax(y_test[2])] model.save('fin_mod.h5')
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score yhat = model.predict(X_test)

 
ytrue = np.argmax(y_test, axis=1).tolist() yhat = np.argmax(yhat, axis=1).tolist() multilabel_confusion_matrix(ytrue, yhat) accuracy_score(ytrue, yhat)
yhat_t = model.predict(X_train)
ytrue_t = np.argmax(y_train, axis=1).tolist() yhat_t = np.argmax(yhat_t, axis=1).tolist() multilabel_confusion_matrix(ytrue_t, yhat_t) accuracy_score(ytrue_t, yhat_t)
# render probabilites
colors	=	[(245,	117,	16),	(117,	245,	16),	(16,	117,	245),	(255,0,0),
(255,153,0),(255,255,0),(128,0,128),(0,255,255),(128,0,0),(127,255,212),(75,0,130
),(250,0,0)]
def prob_viz(res, actions, input_frame, colors): output_frame = input_frame.copy()
for num, prob in enumerate(res):
cv2.rectangle(output_frame, (0, 60+num*35), (int(prob*100), 90+num*35), colors[num], -1) # drawing a dynamic rectangle
cv2.putText(output_frame,	actions[num],	(0,	85+num*35), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
return output_frame
#1. new detection variables sequence = []
sentence = [] predictions = [] threshold = 0.7


 
cap = cv2.VideoCapture(0) #to access webcam device

# set mediapipe model
with	mp_holistic.Holistic(min_detection_confidence	=	0.5, min_tracking_confidence = 0.5) as holistic:
while cap.isOpened():	# loop through all frames


# read frames
ret, frame = cap.read()

# make detections
image, results = mediapipe_detection(frame, holistic)

# draw formatted landmarks draw_styled_landmarks(image, results)
#
#2. prediction logic
keypoints = extract_keypoints(results) # extracting the keypoints sequence.append(keypoints)
sequence = sequence[-10:] #grab last 10 frames


if len(sequence) == 10:
res = model.predict(np.expand_dims(sequence, axis=0))[0] #	print(actions[np.argmax(res)])
predictions.append(np.argmax(res))
#
#3. visualisation logic

 
if np.unique(predictions[-10:])[0]==np.argmax(res): if res[np.argmax(res)] > threshold:
if len(sentence) > 0: #check for words in sentence if actions[np.argmax(res)] != sentence[-1]:
sentence.append(actions[np.argmax(res)])
else:
sentence.append(actions[np.argmax(res)])

if len(sentence) > 8: sentence = sentence[-8:]
#	q
#4. visualization probabilites
image = prob_viz(res, actions, image, colors) #	q

cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)

# show image on screen
cv2.imshow('OpenCV Feed',cv2.resize(image,(600,500)))


# break gracefully if hit 'q' on keyboard if cv2.waitKey(10) & 0xFF == ord('q'):
break

 
cap.release()	# release webcam cv2.destroyAllWindows() # close down all frames
