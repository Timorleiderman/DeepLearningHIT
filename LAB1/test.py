import tensorflow
from keras.datasets import mnist
from keras.models import Sequential, Model
import keras.layers as layers
from keras.utils import to_categorical

from keras import optimizers
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Change labels to one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Print shapes
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

num_of_clss = 10           # number of classes
hidden_size = 10           # number of neurons in the hidden layer
lr = 1e-3 #usually start from 10^-3                  # learning rate
beta_1 =  0.9 #momentum speed usually 0.9               # beta 1 - for adam optimizer
beta_2 =  0.95 # divide the values in squre grads usually 0.99              # beta 2 - for adam optimizer
epsilon = 1e-7        # epsilon - for adam optimizer
epochs = 20                # number of epochs how many times iterate over the data
bs = 32  #usually multiply by 2 16 32 64 128 # bach calc loss update params                 # batch size


#from tensorflow.python.keras import backend as k
from tensorflow.keras.models import Sequential
#from keras.layers import Activation, Dense ,Flatten
from tensorflow.keras.layers import Flatten, Dropout, Activation, Input, Dense, concatenate

in_dim = 28*28

model = Sequential()
model.add(Flatten())
model.add(Dense(hidden_size, input_dim=in_dim))
model.add(Activation('sigmoid'))
model.add(Dense(num_of_clss, input_dim=in_dim))
model.add(Activation('softmax'))

# define the optimizer and compile the model

adam = optimizers.Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model, iterating on the data in batches
history = model.fit(x_train, y_train, validation_split=0.3, epochs=epochs, batch_size=bs)

# plot train and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show(); plt.close()

test_loss, test_acc = model.evaluate(x_test, y_test)

# Print results
print('test loss:', test_loss)
print('test accuracy:', test_acc)

y_pred = model.predict(x_test)

# Confusion Matrix
cm = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(np.round(y_pred),axis=1))
labels = ['class ' + str(i) for i in range(num_of_clss)]
plot_confusion_matrix(cm,labels,title='Confusion Matrix',normalize=True)

# Summerize the model arhiteture and parameters
model.summary()


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

from tensorflow.keras import models

loaded_model = tensorflow.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

y_pred = loaded_model.predict(x_test)

# Confusion Matrix
cm = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(np.round(y_pred),axis=1))
labels = ['class ' + str(i) for i in range(num_of_clss)]
plot_confusion_matrix(cm,labels,title='Confusion Matrix',normalize=True)

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Plot linewidth.
lw = 2

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_of_clss):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot all ROC curves
colors = ['aqua', 'darkorange', 'cornflowerblue', 'red', 'green', 'blue', 'black', 'brown', 'purple', 'pink']
for i, color in zip(range(num_of_clss), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (AUC = {2})' ''.format(i, roc_auc[i],roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

