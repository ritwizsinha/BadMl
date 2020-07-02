from keras.models import model_from_json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelBinarizer

label_binrizer = LabelBinarizer()
from string import ascii_uppercase
count = 0
map = dict()
for i in ascii_uppercase:
    if( i != 'J' and i !='Z'):
        map[count] = i
        count+=1
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
test = pd.read_csv('../Desktop/sign_mnist_test.csv')
test_labels = test['label']
test.drop('label', axis = 1, inplace = True)
test_images = test.values
test_images = np.array([np.reshape(i, (28, 28)) for i in test_images])
test_images = np.array([i.flatten() for i in test_images])
test_labels = label_binrizer.fit_transform(test_labels)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
print(test_images.shape)
y_pred = loaded_model.predict(test_images)
# for i in range(7172):
#     for j in range(24):
#             print(str(y_pred[i][j]) + " ")
#     print("\n")
from sklearn.metrics import accuracy_score
print(accuracy_score(test_labels, y_pred.round()))