
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras 
from matplotlib import pyplot as plt

def preprocess (img,label):
    return tf.image.resize(img,[200, 200])/255, label

#split = ["train[:70%]","train[70:]"]

#name = setattr(tfds.image_classification.cats_vs_dogs, '_URL',"https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip")

trainDataset, testDataset = tfds.load(name = 'cats_vs_dogs',split = ["train[:70%]","train[70%:]"] ,as_supervised = True)
trainDataset = trainDataset.map(preprocess).batch(32)
testDataset = testDataset.map(preprocess).batch(32)

model = keras.Sequential([
    keras.layers.Conv2D(16,(3,3),activation = 'relu',input_shape = (200, 200,3)),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(32,(3,3),activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64,(3,3),activation = 'relu'),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation = 'relu'),
    keras.layers.Dense(1,activation = 'sigmoid'),
])

model.compile(optimizer = "adam",loss='binary_crossentropy', metrics = ['accuracy'])

trainHistory = model.fit(trainDataset, epochs = 10, validation_data = testDataset)

plt.plot(trainHistory.history['accuracy'])
plt.plot(trainHistory.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Trainung', 'Validation'])
plt.grid()
plt.show()

(loss, accuracy) = model.evaluate(testDataset)
print(loss)
print(accuracy)

model.save("catdog_mod.h5")


model = keras.models.load_model("catdog_mod.h5")
predictions = model.predict(testDataset.take(10))

classNames = ['cat','dog']

i = 0
fig,ax = plt.subplots(1,10)
for image, _ in testDataset.take(10):
    predictedLabel = int(predictions[i] >= 0.5)
    ax[i].axis('off')
    ax[i].set_title(classNames[predictedLabel])
    ax[i].imshow(image[i])
    i += 1

plt.show()