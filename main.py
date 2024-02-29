# Import nescessary libraries for the project
import os
import cv2  # To load and process images
import numpy as np  # Working with numpy-arrays
import matplotlib.pyplot as plt
import tensorflow as tf  # For all the machinelearning

# Load datset
mnist_dataset = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()

# Normalize pixels of the image
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

"""
# Following outcommented code, was used train and save the model. 
# This allows for not training the model every time the code is executed.
# Create model for the neural network
nw_model = tf.keras.models.Sequential()

# Add layers to the model
# First layer (input-layer) turns the 28x28 grid into one 'line' containing the same amount of pixels.
nw_model.add(tf.keras.layers.Flatten(input_shape = (28, 28))) 

# Second and third layer connects neurons of the other layers
nw_model.add(tf.keras.layers.Dense(128, activation = 'relu'))
nw_model.add(tf.keras.layers.Dense(128, activation = 'relu'))

# The ouputlayers repsective units represents the given digit from the input-image
nw_model.add(tf.keras.layers.Dense(10, activation = 'softmax')) # The activation function 'softmax' makes sure all neurons in this layer adds up to 1'

# Compile model 
nw_model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model by passing the trainingdata
nw_model.fit(x_train, y_train, epochs = 3) # Ephocs specifies how many times the model gets shown the trainingdata  

nw_model.save('handwritten-digits.model')
"""

""""
# Test how well the model performs on testingdata
loss, accuracy = nw_model.evaluate(x_test, y_test)
print(loss)
print(accuracy)
"""

# Load the newly created model
model = tf.keras.models.load_model("handwritten-digits.model")

# Read digit-files
img_num = 1
while os.path.isfile(f"digits_written_on_paper/digit{img_num}.png"):
    try:
        input_image = cv2.imread(
            f"digits_written_on_paper/digit{img_num}.png", cv2.IMREAD_GRAYSCALE
        )  # Only take white and black colours of the image
        # By default the image returns white on black, and not black on white
        input_image = np.invert(
            np.array([input_image])
        )  # Note that the input-image should be converted as list in a np-array in order for the model to load the data
        digit_prediction = model.predict(input_image)
        print(
            f"Digit represented on the image should be: {np.argmax(digit_prediction)}"
        )  # argmax returns the index of the field that has the highest activation in the last layer (representing the digit predicted)
        plt.imshow(input_image[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("An error has occurred.")
    finally:
        img_num += 1
