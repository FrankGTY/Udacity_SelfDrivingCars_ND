import csv
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from scipy import ndimage
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout

LOG_DIR = '/opt/carnd_p3/data/'
BATCH_SIZE = 32

# define a generator for the model
def generator(logs, batch_size=32):
    num_logs = len(logs)
    angle_correction = 0.2
    while 1: 
        shuffle(logs)
        for offset in range(0, num_logs, batch_size):
            # iterate through the logs in batches
            batch_logs = logs[offset:offset+batch_size]
            images, angle_measurements = [], []
            for log in batch_logs:
                # reformat the paths so that the filenames are accurate for this workspace
                center_path = LOG_DIR+'IMG/'+log[0].split('/')[-1]
                left_path = LOG_DIR+'IMG/'+log[1].split('/')[-1]
                right_path = LOG_DIR+'IMG/'+log[2].split('/')[-1]
                # capture the images and steering angles for each of the three cameras
                center_image, left_image, right_image = ndimage.imread(center_path), ndimage.imread(left_path), ndimage.imread(right_path)
                center_angle = float(log[3])
                left_angle, right_angle = center_angle+angle_correction, center_angle-angle_correction
                # add the images and angles to the data
                images.extend([center_image, left_image, right_image])
                angle_measurements.extend([center_angle, left_angle, right_angle])
                # augment the data by flipping the image and angles
                images.extend([cv2.flip(center_image, 1), cv2.flip(left_image, 1), cv2.flip(right_image, 1)])
                angle_measurements.extend([-1.0*center_angle, -1.0*left_angle, -1.0*right_angle])
            # set up the training data and label arrays
            X_train = np.array(images)
            y_train = np.array(angle_measurements)
            yield shuffle(X_train, y_train)

# read driving logs to an array
driving_logs = []
with open(LOG_DIR+'driving_log.csv') as driving_log_file:
    reader = csv.reader(driving_log_file)
    for line in reader:
        driving_logs.append(line)
        
# split the logs into training and validation data
train_logs, validation_logs = train_test_split(driving_logs, test_size=0.2)

# set up the data via generator functions
train_generator = generator(train_logs, batch_size=BATCH_SIZE)
validation_generator = generator(validation_logs, batch_size=BATCH_SIZE)

model = Sequential()
model.add(Lambda(lambda x: x/255-0.5, input_shape=(160,320,3)))  # data noamalization
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dropout(0.5)) # adding in a single dropout layer to reduce overfitting
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, 
          steps_per_epoch=np.ceil(len(train_logs)/BATCH_SIZE),
          validation_data=validation_generator,
          validation_steps=np.ceil(len(validation_logs)/BATCH_SIZE),
          epochs=5, verbose=1)   
model.save('model.h5') 

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_logs), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_logs), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
exit()