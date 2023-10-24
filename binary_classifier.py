'''
Steps of an ML script:-
1. Load data (using pandas or whatever)
2. Split data into training and test sets
3. build a model using Tensorflow
4. Fit the model model.fit(x, y)
5. Evaluate the model
6. Make predictions with the model
'''
'''
NOTE:-
THIS IS ALSO CALLED LOGISTIC REGRESSION
'''
import tensorflow as tf

#################################### STEP 1: LOAD DATA ######################################################################################
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
print(data.data.shape) #gives you an overview of the entire dataset, here, the result is (569, 30). (N(no. of samples), D(no. of features))
#the way to access this data is below:-
#print(data)

#print(data.keys()) #for keys
#access data through keys like below:-
#print(data.target) # or print(data.frame), print(data.filename) etc etc....

#################################### STEP 2: SPLIT DATA ######################################################################################
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33)

#NOTE: PREPROCESSING NUMERIC DATA IS DONE BY SCALING. THIS MAKES ALL THE DATA SIMILAR TO EACHOTHER SO THAT NO FEATURE DONMIATES THE OTHER DUE TO ITS SCALE
#preprocessing data by scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# 'fit' calculates the mean and SD for each feature in training data, 'transform' does the actual scaling wrt the computed mean and SD
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

N,D = X_train.shape
#################################### STEP 3: INITIATE ML MODEL ###############################################################################
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(D,)),
    tf.keras.layers.Dense(1, activation='sigmoid') #here, the '1' argument means one output node or one output neuron. the reson why we picked 1 is because there can only be one output in binary classifiers. (either the patient has or does not have cancer)
])

#we know that we need to minimise loss to gain better accuracy. how is that done? => using optimizer. optimizer is the part that finds patterns in the data.
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

#################################### STEP 4: FIT MODEL ######################################################################################
r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

#################################### STEP 5: EVALUATE MODEL #################################################################################
print()
print("Train score:",model.evaluate(X_train, y_train))
print("Test Score:",model.evaluate(X_test, y_test))

predictions = model.predict(X_test)
threshold = 0.5
bin_predictions = (predictions > threshold).astype(int)
print(bin_predictions)

class_labels = ["Benign", "Malignant"]
for i, prediction in enumerate(bin_predictions):
    data_point = X_test[i]
    print("Data Point:", data_point)
    print("Prediction:", prediction)
