import csv
import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import GlorotNormal
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, l2
from keras.utils import to_categorical
from matplotlib import pyplot as plt
from random import randint

lr = 0.001 # VAR: learning rate
m = 0.2 # VAR: momentum constant

data = np.loadtxt("data/dataset-normalized.csv", delimiter=";")

X = data[:, 6:-1]
Y = data[:, -1]

initializer = GlorotNormal(seed=randint(0, 100000))
model = Sequential()
model.add(Dense(23, activation="relu", input_dim=12, kernel_initializer=initializer, bias_initializer=initializer))
model.add(Dense(23, activation="relu", kernel_initializer=initializer, bias_initializer=initializer))
model.add(Dense(11, activation="relu", kernel_initializer=initializer, bias_initializer=initializer))
model.add(Dense(5, activation="softmax", kernel_initializer=initializer, bias_initializer=initializer))

keras.optimizers.SGD(learning_rate=lr, momentum=m)
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=['mse', 'accuracy'])

cb = [EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=8, start_from_epoch=15, restore_best_weights=True)]
model.fit(X, to_categorical(Y), epochs=100, batch_size=100, callbacks=cb)

def model_eval(array):
    return model.predict(np.array([array]))

print(f"Evaluation for class sitting: {model_eval([0.36670501704077363, 0.41913818751633, 0.5458295015133863, 0.44993109399910874, 0.5687536588742678, 0.7169090890285302, 0.5192832262022264, 0.58111748019563, 0.5059740406471116, 0.8276743434002224, 0.6800047776149954, 0.7591477337378798])}")
print(f"Evaluation for class sittingdown: {model_eval([0.37314214485831265, 0.4571115842113521, 0.48084340164391676, 0.41401149578956675, 0.5558311719739546, 0.6083430118443007, 0.5044307588139885, 0.5984389976676625, 0.5111380499298249, 0.7531576110012722, 0.7025114518623886, 0.8019672272563738])}")
print(f"Evaluation for class standing: {model_eval([0.36749517892689193, 0.45864229622417785, 0.4893691402484699, 0.49165300602814566, 0.7029952080415054, 0.6387894166871629, 0.5188892819985302, 0.6003160164021469, 0.5132187263139888, 0.7602577737007218, 0.7200672704519145, 0.7685902350263539])}")
print(f"Evaluation for class standingup: {model_eval([0.36881573446168536, 0.45066134608093733, 0.4762639941098505, 0.4048881329399422, 0.5442685082232757, 0.5935697681237714, 0.4889878547308064, 0.5502809475821215, 0.49679735977464345, 0.7350129997430215, 0.708397697729003, 0.7755384576248966])}")
print(f"Evaluation for class walking: {model_eval([0.3657606499476145, 0.46254931839993396, 0.4808155743536104, 0.31293084320011405, 0.44732838943844183, 0.41328000034928825, 0.5097555024771858, 0.6240887339520145, 0.5049057210771296, 0.7491501288980927, 0.7293664996151338, 0.7504830494162258])}")

print(f"Evaluation for 20,0.6,0.00: {model_eval([0.332, 0.508, 0.757, 0.364, 0.823, 0.652, 0.691, 0.535, 0.495, 0.799, 0.673, 0.374])}")
print(f"Evaluation for 20,0.6,0.01: {model_eval([0.348, 0.5, 0.533, 0.35, 0.621, 0.786, 0.469, 0.485, 0.494, 0.784, 0.724, 0.763])}")
print(f"Evaluation for 20,0.6,0.1: {model_eval([0.396, 0.343, 0.49, 0.583, 0.662, 0.785, 0.368, 0.413, 0.61, 0.718, 0.626, 0.697])}")
print(f"Evaluation for 20,0.9,0.01: {model_eval([0.345, 0.394, 0.529, 0.505, 0.567, 0.684, 0.515, 0.569, 0.472, 0.748, 0.669, 0.806])}")
print(f"Evaluation for 20,0.1,0.01: {model_eval([0.371, 0.362, 0.433, 0.524, 0.534, 0.75, 0.482, 0.569, 0.502, 0.776, 0.702, 0.751])}")

print(f"Evaluation for 200,0.6,0.00: {model_eval([0.314, 0.417, 0.586, 0.494, 0.596, 0.727, 0.462, 0.619, 0.537, 0.839, 0.718, 0.753])}")
print(f"Evaluation for 200,0.6,0.01: {model_eval([0.38, 0.441, 0.553, 0.449, 0.574, 0.75, 0.537, 0.551, 0.494, 0.838, 0.683, 0.757])}")
print(f"Evaluation for 200,0.6,0.1: {model_eval([0.401, 0.443, 0.618, 0.481, 0.401, 0.638, 0.572, 0.543, 0.453, 0.892, 0.792, 0.693])}")
print(f"Evaluation for 200,0.9,0.01: {model_eval([0.379, 0.44, 0.523, 0.434, 0.554, 0.715, 0.542, 0.58, 0.497, 0.819, 0.685, 0.755])}")
print(f"Evaluation for 200,0.1,0.01: {model_eval([0.376, 0.427, 0.554, 0.516, 0.571, 0.696, 0.518, 0.608, 0.492, 0.819, 0.683, 0.749])}")
