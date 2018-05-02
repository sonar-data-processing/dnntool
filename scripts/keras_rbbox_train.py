import os
import time
import keras_rbbox_util as util
import keras
import keras.layers as KL
from keras.models import Model
from keras.callbacks import ModelCheckpoint

image_input = KL.Input(shape=(224, 224, 3))
model = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=image_input)
x = KL.Flatten(name='flatten')(model.output)
x = KL.Dense(1024, init='normal', activation='relu') (x)
x = KL.Dense(1024, init='normal', activation='relu')(x)
out = KL.Dense(4, init='normal', name='out_bboxes_poses')(x)

model = Model(input=model.input, output=out)

if os.path.isfile("rbbox-weights.hdf5"):
    model.load_weights("rbbox-weights.hdf5")

model.summary()

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

# Loading the training and validation data
X_train, y_train, X_val, y_val, X_test, y_test = util.load_data("/home/gustavoneves/data/gemini/dataset/bbox")

t = time.time()

checkpointer = ModelCheckpoint(filepath="rbbox-weights.hdf5", monitor="val_loss", verbose=1, save_best_only=True)
hist = model.fit(X_train, y_train, batch_size=32, epochs=10000, verbose=1, validation_data=(X_val, y_val), callbacks=[checkpointer])

print('Training time: %s' % (time.time()-t))

(loss, accuracy) = model.evaluate(X_test, y_test, batch_size=1, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))
