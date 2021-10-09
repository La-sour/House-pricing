# House-pricing

In this exercise you'll try to build a neural network that predicts the price of a house according to a simple formula.

So, imagine if house pricing was as easy as a house costs 50k + 50k per bedroom, so that a 1 bedroom house costs 100k, a 2 bedroom house costs 150k etc.

How would you create a neural network that learns this relationship so that it would predict a 7 bedroom house as costing close to 400k etc.

Hint: Your network might work better if you scale the house price down. You don't have to give the answer 400...it might be better to create something that predicts the number 4, and then your answer is in the 'hundreds of thousands' etc.

This is written using phyton and tensorflow. Then, we import keras. In keras we use the word dense to define a layer of connected neurons. Numpy that makes data representations particularly enlist much easier.


import tensorflow as tf 
import numpy as np
from tensorflow import keras

we have one list for x and y. here, we are asking the model to figure out how to fit the x values to the y values. it's a single neuron because there is only one layer and one dense layer.

# GRADED FUNCTION: model_house
def model_house(y_mod):
    x = np.array([ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    y = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])]) # Your Code Here#
    model.compile(optimizer='sgd', loss="mean_squared_error")
    model.fit(x, y, epochs=500)
    return model.predict(y_mod)[0]
    
This code defines them. So it make a guess, the loss function measures this and then gives the data to the optimizer which figure out the next guess.

prediction = model_house([7.0])
print(prediction)
