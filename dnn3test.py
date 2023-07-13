import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import Normalizer
from keras import callbacks
from keras.callbacks import ModelCheckpoint, CSVLogger

np.random.seed(1337)  # for reproducibility

traindata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Training.csv', header=None)
testdata = pd.read_csv('https://raw.githubusercontent.com/rahulvigneswaran/Intrusion-Detection-Systems/master/dnn/kdd/binary/Testing.csv', header=None)

X = traindata.iloc[:, 1:42]
Y = traindata.iloc[:, 0]
C = testdata.iloc[:, 0]
T = testdata.iloc[:, 1:42]

scaler = Normalizer().fit(X)
trainX = scaler.transform(X)

scaler = Normalizer().fit(T)
testT = scaler.transform(T)

y_train = np.array(Y)
y_test = np.array(C)

X_train = np.array(trainX)
X_test = np.array(testT)

batch_size = 64

# Define the network
model = Sequential()
model.add(Dense(1, input_dim=41, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="kddresults/dnn3layer/checkpoint-{epoch:02d}.hdf5", verbose=1, save_best_only=True, monitor='loss')
csv_logger = CSVLogger('kddresults/dnn3layer/training_set_dnnanalysis.csv', separator=',', append=False)
model.fit(X_train, y_train, batch_size=batch_size, callbacks=[checkpointer, csv_logger])
model.save("kddresults/dnn3layer/dnn3layer_model.hdf5")

score = []

y_train1 = y_test
y_pred = model.predict(X_test)

threshold = 0.5
y_pred_thresholded = (y_pred >= threshold).astype(int)

accuracy = accuracy_score(y_train1, y_pred_thresholded)
recall = recall_score(y_train1, y_pred_thresholded, average="binary")
precision = precision_score(y_train1, y_pred_thresholded, average="binary")
f1 = f1_score(y_train1, y_pred_thresholded, average="binary")
print("----------------------------------------------")
print("accuracy")
print("%.3f" % accuracy)
print("recall")
print("%.3f" % recall)
print("precision")
print("%.3f" % precision)
print("f1score")
print("%.3f" % f1)
score.append(accuracy)

# Get the weights of the first layer
weights = model.layers[0].get_weights()[0].flatten()

# Check if the length of weights matches the number of features
if len(weights) != X_train.shape[1]:
    print("Error: The number of weights does not match the number of features.")
else:
    feature_names = [
        "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
        "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
        "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
        "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
        "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
        "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
        "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
    ]

    # Create a DataFrame for features and their corresponding weights
    features_table = pd.DataFrame({
        'Feature': feature_names,
        'Weight': weights
    })

    # Sort the features table by weight in descending order
    features_table = features_table.sort_values(by='Weight', ascending=False)

    # Print the features table
    print("Features Table:")
    print(features_table.to_string(index=False))

