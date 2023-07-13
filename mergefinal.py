import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from keras.models import load_model
from keras.models import Model
from keras.models import save_model
from sklearn.preprocessing import LabelEncoder

# Load the trained models
model1 = load_model('kddresults/dnn3layer/dnn3layer_model.hdf5')
model2 = load_model('kddresults/testing/test_model.hdf5')

# Load the kddcup99 dataset
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

# Prepare the dataset for evaluation
X_test = testdata.iloc[:, :-1].values  # Features
y_test = testdata.iloc[:, -1].values   # Labels

# Make predictions using the reference models
y_pred_model1 = np.argmax(model1.predict(X_test), axis=-1)
y_pred_model2 = np.argmax(model2.predict(X_test), axis=-1)

# Create a new IDS model using the predictions of the reference models
# Assuming a majority voting approach
y_pred_new = np.round((y_pred_model1 + y_pred_model2) / 2).astype(int)

# Encode labels if necessary
label_encoder = LabelEncoder()
label_encoder.fit(y_test)

# Handle unseen labels
unique_labels_test = np.unique(y_test)
unique_labels_pred = np.unique(y_pred_new)
unseen_labels = np.setdiff1d(unique_labels_pred, unique_labels_test)

missing_label = -1  # Replace with a valid label value
label_encoder.classes_ = np.append(label_encoder.classes_, missing_label)

# Map unseen labels to missing label
y_pred_new = np.where(np.isin(y_pred_new, unseen_labels), missing_label, y_pred_new)

# Encode the labels
y_test_encoded = label_encoder.transform(y_test)
y_pred_new_encoded = label_encoder.transform(y_pred_new)

# Define the threshold for binary classification
threshold = 0.5

# Apply threshold to obtain binary predictions
y_pred_thresholded = (y_pred_new_encoded >= threshold).astype(int)

# Evaluate the model using the provided reference
accuracy = accuracy_score(y_test_encoded, y_pred_thresholded)
recall = recall_score(y_test_encoded, y_pred_thresholded, average='micro')
precision = precision_score(y_test_encoded, y_pred_thresholded, average='micro')
f1 = f1_score(y_test_encoded, y_pred_thresholded, average='micro')

print("----------------------------------------------")
print("accuracy")
print("%.3f" % accuracy)
print("recall")
print("%.3f" % recall)
print("precision")
print("%.3f" % precision)
print("f1score")
print("%.3f" % f1)

