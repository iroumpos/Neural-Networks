# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from rbflayer import RBFLayer, InitCentersRandom
import matplotlib.pyplot as plt





# # RBF Network Prediction <br>
# # MSE loss plot over centers



# +


# Generate random data points
np.random.seed(42)
### Gaussian distribution of data points ####
#X_normal = np.random.normal(0, 2, size=(30, 1))
#X_normal = np.clip(X_normal, -4, 4)  


#### Uniform distribution of data points ###
X = np.random.uniform(low=-4, high=4, size=(30, 1))
y = 1 + np.sin(X * np.pi / 8)

num_centers = [4,8, 12, 20]
mse = {}
histories = []

for num_centers in num_centers:
    
    model = Sequential()
    rbflayer = RBFLayer(num_centers, initializer=InitCentersRandom(X),betas=0.5, input_shape=(1,))
    model.add(rbflayer)
    model.add(Dense(1))
    sgd_optimizer = SGD(learning_rate=0.01)
    model.compile(loss='mean_squared_error', optimizer=sgd_optimizer)
    
    ### Initial weights and biases###
    initial_centers = model.layers[0].get_weights()[0]
    initial_betas = model.layers[0].get_weights()[1]
    print("Initial Centers:\n", initial_centers)
    print("\nInitial Betas:\n", initial_betas)
    
    
    history = model.fit(X, y, epochs=500, verbose=0)
    histories.append(history)
    ### Get mse convergence for every number of centers ####
    mse[f'RBF Centers: {num_centers}'] = history.history['loss'][499]
    
    ### Position of weights - centers and biases after training ###
    trained_centers = model.layers[0].get_weights()[0]
    trained_betas = model.layers[0].get_weights()[1]

    print("\nTrained Centers:\n", trained_centers)
    print("\nTrained Betas:\n", trained_betas)
    
    
    
    
    X_test = np.linspace(-4, 4, 100).reshape(-1, 1)
    y_pred = model.predict(X_test)

    plt.figure()
    plt.scatter(X, y, label='True data')
    plt.plot(X_test, y_pred, label='RBF Network Prediction', color='red')
    plt.xlabel('p')
    plt.ylabel('g(p)')
    plt.legend()
    plt.title(f'RBF Network Prediction - Centers: {num_centers}')
    plt.show()
    
    
    
##### MSE Loss #####    
plt.figure(figsize=(10, 6))
labels = [4,8,12,20]
for i, history in enumerate(histories):
    plt.plot(history.history['loss'], label=f'RBF Centers: {labels[i]}')

plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training Loss for Different Numbers of Centers')
plt.legend()
plt.show()



# -



# # MSE convergence

mse



# Sampling data points with uniform distribution in interval [-4,4] which means that all values within a given range have an equal probability of occurring <br>
# Weights are initialized also random in the interval (-4,4) <br> 
# Betas are initialized with value 1 <br>
# Learning rate = 0.01 <br>
# MSE is low for every possible choice of number of centers. As centers are increased the loss is decreased. <br> 
# More significant change is from 4 centers to 8 and from 8 to 12. Changing the number of centers from 12 to 20 does not decrease so much the mse 





# # RBF
# Centers: Each center represents a specific location in the input space. This location determines the activation strength of the corresponding basis function in the network. <br>
# More centers means more flexibility since they can capture better the input space <br>
# Biases:The bias (std dev OR variance OR spread const) performs a
# scaling operation on the transfer (basis) function, causing it to stretch or compress
#



# ### 4 Centers <br>
# curve does not fit well the data, only in regioin [-1,1] 4 centers cannot capture the input space <br>
# ### 8 Centers <br>
# Curve captures well the pattern of the function that we try to approximate except the edges of the interval.Also the curve is smooth and not strictly passing through the data points providing a flexibility. <br>
# ### 12 Centers <br>
# Curve captures even better the pattern of the function and in compare to 8 Centers, in p > 2 the curve is getting very closer to the data points from the curve of 8 Centers <br>
# ### 20 Centers <br>
# Curve is even more accurate and in compare to 12 Centers, in p < -2 the curve is getting closer to the data points 





# # Try different sets of initial weights, different sampling methods for selecting training pairs

# Sampling data points with Gaussian distribution in interval [-4,4]  with mean = 0 and std = 1 <br>
# Weights are initialized also random in the interval (-4,4) <br> 
# Betas are initialized with value 0.5 <br>
# Learning rate = 0.01 <br>
# MSE is low for every possible choice of number of centers. As centers are increased the loss is decreased. <br> 
# More significant change is from 4 centers to 8 and from 8 to 12. Changing the number of centers from 12 to 20 does not decrease so much the mse 



# Lowering the values of biases has as result the centers to be wider and capture more points in the function activation region

#
#
# ### 4 Centers <br>
# curve fits the data points significant better than the curve with the previous initializations. It does not fit well after p > 2. <br>
# ### 8 Centers <br>
# Curve captures better the patterns of the function in regions p < -2 and p > 2 in compare to the previous initializations. <br>
# ### 12 Centers <br>
# Curve captures even better the pattern of the function and in compare to 8 Centers, in p > 2 the curve is getting very closer to the data points from the curve of 8 Centers <br>
# ### 20 Centers <br>
# Curve is even more accurate and in compare to 12 Centers and in compare to the previous initializations. It fits almost perfect the data points and approximate very efficient the function 










