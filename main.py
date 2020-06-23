import  tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# we load images data from mnist

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# CHange data to the correct format

train_images = train_images / 255.0
test_images = test_images / 255.0

num_classes = 10
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

# Our model for classification

model  = keras.Sequential([
                           keras.layers.Flatten(input_shape = (28,28)),
                           keras.layers.Dense(128, activation='relu'),
                           keras.layers.Dense(num_classes,activation = 'softmax')
])

model.compile(loss=tf.keras.losses.categorical_crossentropy,optimizer = keras.optimizers.SGD(lr= 0.3),metrics=['accuracy'])

#We dont want to add more layer because it will cause our model to overfit

model.fit(train_images,train_labels,epochs = 20)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

predictions = model.predict(test_images)

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

print("Our prediction for the second image is " + class_names[np.argmax(predictions[1])])

