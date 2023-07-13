import pandas as pd
from keras.utils import get_file
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt

# Download and read the dataset
try:
    path = get_file('kddcup.data_10_percent.gz', origin='https://figshare.com/ndownloader/files/5976042')
except:
    print('Error downloading the dataset.')
    raise

df = pd.read_csv(path, header=None)

df.columns = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
    'land',
    'wrong_fragment',
    'urgent',
    'hot',
    'num_failed_logins',
    'logged_in',
    'num_compromised',
    'root_shell',
    'su_attempted',
    'num_root',
    'num_file_creations',
    'num_shells',
    'num_access_files',
    'num_outbound_cmds',
    'is_host_login',
    'is_guest_login',
    'count',
    'srv_count',
    'serror_rate',
    'srv_serror_rate',
    'rerror_rate',
    'srv_rerror_rate',
    'same_srv_rate',
    'diff_srv_rate',
    'srv_diff_host_rate',
    'dst_host_count',
    'dst_host_srv_count',
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate',
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
    'outcome'
]

# Drop duplicate rows and missing values
df.drop_duplicates(keep='first', inplace=True)
df.dropna(inplace=True)

# Encode labels
encoder = LabelEncoder()
df['outcome'] = encoder.fit_transform(df['outcome'])

# Separate features and labels
x = df.drop('outcome', axis=1)
y = df['outcome']

# Perform one-hot encoding for categorical variables
cat_features = ['protocol_type', 'service', 'flag']
preprocessor = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_features)], remainder='passthrough')
x = preprocessor.fit_transform(x)

# Select the relevant columns
x = x[:, :41]  # Select the first 41 columns

# Split the dataset into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)

# Scale the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Create the model
model = Sequential()
model.add(Dense(64, input_dim=41, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(len(encoder.classes_), activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
history = model.fit(x_train, y_train, validation_split=0.2, epochs=5, batch_size=41, callbacks=[early_stopping])

# Evaluate the model on the test set
y_pred = np.argmax(model.predict(x_test), axis=-1)

# Print the classification report
classification_rep = classification_report(y_test, y_pred, zero_division=1)
print(classification_rep)

model.save("kddresults/testing/test_model.hdf5")

# Plot the training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.close()