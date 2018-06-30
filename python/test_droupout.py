import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.applications.resnet50 import ResNet50
from keras.models import Model, Sequential
from keras.layers import (Conv2D, Activation, MaxPool2D, Dense, Flatten, GlobalAveragePooling2D, Average,
                          AveragePooling2D, RepeatVector, Reshape, Input, Dropout, Concatenate, TimeDistributed,
                            )
from keras.optimizers import SGD, Adam
from keras.backend import learning_phase, function
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
def baseline_model():
    inputs = Input(shape=(1,))
    x = Dense(20, activation='relu', kernel_initializer='normal')(inputs)
    x = Dropout(0.05)(x)
    x = Dense(20, activation='relu', kernel_initializer='normal')(x)
    x = Dropout(0.05)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

dataset = np.array([[x, x**2] for x in np.arange(-2, 7, 0.001)])
X_train, y_train = dataset[:2000,0].reshape(-1, 1), dataset[:2000,1].reshape(-1, 1)
X, y = dataset[:,0], dataset[:,1]

plt.scatter(X, y, s=0.5, label='Neural Net Prediction', color="red")
plt.plot(dataset[:,0], dataset[:,1])

seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=10, batch_size=5, verbose=0)

np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=1, verbose=1)))
pipeline = Pipeline(estimators)

model = baseline_model()

if os.path.isfile("dropout.h5"):
    model.load_weights("dropout.h5")
else:
    model.fit(X_train, y_train, epochs=2000, batch_size=10)
    model.save_weights("dropout.h5");

test_batch_size = 9000
# Setup a Keras fucntion to use dropout vational inference in test time
get_dropout_output = function([model.layers[0].input, learning_phase()], [model.layers[-1].output])
mc_dropout_num = 100 # Run Dropout 100 times
predictions = np.zeros((mc_dropout_num, test_batch_size, 1))
uncertainty = np.zeros((mc_dropout_num, test_batch_size, 1))
for i in range(mc_dropout_num):
    result = get_dropout_output([X.reshape((X.shape[0], 1)), 1])[0]
    predictions[i] = result

# get mean results and its varience
prediction_mc_droout = np.mean(predictions, axis=0)
std_mc_droout = np.std(predictions, axis=0)*100

# Array for the real equation
x_true = X
y_true = y

plt.figure(figsize=(10, 7), dpi=100)
plt.errorbar(X, prediction_mc_droout, yerr=std_mc_droout, markersize=2, fmt='o', ecolor='g', capthick=2, elinewidth=0.5,
             label='Neural Ne1t Prediction with epistemic uncertainty by vational inference (dropour)')
plt.axvline(x=1.0, label="Training Data range (0.0 to 1.0)")
plt.plot(x_true, y_true, color='red', label='Real Answer')
plt.xlabel('Data')
plt.ylabel('Answer')
plt.legend(loc='best')
plt.show()

for i in range(1000):
    print(pipeline.predict(X[0:1,:]))
