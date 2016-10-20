from keras.models import Sequential
from keras.layers import Dense
import numpy as np

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load pima indians dataset
dataset = np.loadtxt("pima-indians-diabetes.csv", delimiter=",")

# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = Sequential()
model.add(Dense(output_dim=12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(output_dim=8, init='uniform', activation='relu'))
model.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit model
model.fit(x=X, y=Y, nb_epoch=150, batch_size=5)

#evaluate the model
score = model.evaluate(x=X, y=Y)
print score




