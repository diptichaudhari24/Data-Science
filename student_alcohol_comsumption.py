import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

# READING DATA AND CLASSIFIY TRUE FOR STUDENT DRINKING ABOVE AVERAGE
df = pd.read_csv("/home/sangram/Downloads/student-mat.csv", sep=',')
df['avg']= (5*df['Dalc']+2*df['Walc'])/7
df['consumption']=df['avg']>2.5
del df['Walc']
del df['Dalc']
del df['avg']
df['sex'].replace('F',0,inplace=True)
df['sex'].replace('M',1,inplace=True)
print(df.head(5))

# NETWORK MODELLING

input_units = 8
hidden_units = 15
output_units = 2

x = tf.placeholder(tf.float32, shape=[1, input_units])
y = tf.placeholder(tf.float32, shape=[1, output_units])

weights_input_to_hidden = tf.Variable(tf.random_normal(shape=[input_units, hidden_units]))
biases_of_hidden_layer = tf.Variable(tf.zeros(shape=hidden_units))

weights_hidden_to_output = tf.Variable(tf.random_normal(shape=[hidden_units, output_units]))
biases_of_output_layer = tf.Variable(tf.zeros(shape=output_units))

hidden_layer_output = tf.sigmoid(tf.add(tf.matmul(x,weights_input_to_hidden),biases_of_hidden_layer))

output_layer_output = tf.sigmoid(tf.add(tf.matmul(hidden_layer_output,weights_hidden_to_output),biases_of_output_layer))

network_output = tf.nn.softmax(tf.sigmoid(tf.add(tf.matmul(hidden_layer_output, weights_hidden_to_output), biases_of_output_layer)))

# loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(network_output, y)
loss = tf.reduce_mean(cross_entropy)

# train step
train_op = tf.train.GradientDescentOptimizer(learning_rate=0.8).minimize(loss)

X_data = df[['sex','age','studytime','failures','famrel','goout','health','absences']]
print(X_data.head(5))
y_data = df['consumption']
print(y_data.head(5))

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1)
sss.get_n_splits(X=X_data, y=y_data)

for train_index, test_index in sss.split(X_data, y_data):
    X_train, X_test = X_data.iloc[train_index], X_data.iloc[test_index]
    y_train, y_test = y_data[train_index], y_data[test_index]


def one_hot(label):
    if label:
        return (np.asarray([[1.0, 0]]))
    else:
        return (np.asarray([[0, 1.0]]))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())


n_epochs = 100
print("Training the network....")
for epoch in range(n_epochs):
    epoch_loss = []
    for i in range(X_train.shape[0]):
        x_instance = np.asarray([X_train.iloc[i]])
        y_instance = one_hot(y_train.iloc[i])
        _, loss_value = sess.run([train_op, loss], feed_dict={x:x_instance, y:y_instance})
        epoch_loss.append(loss_value)
    print ('Loss for ' + str(epoch) + ' epoch: ' + str(np.mean(epoch_loss)))

"""
class_labels = ['True', 'False']

def predictions(x_test, y_test):
    preds = []
    obs   = []
    for i in range(x_test.shape[0]):
        x_instance = np.asarray([x_test.iloc[i]])
        y_instance = one_hot(y_test.iloc[i])
        preds.append(np.argmax(sess.run(network_output, feed_dict={x:x_instance})[0]))
        obs.append(np.argmax(y_instance[0]))
    return (pd.DataFrame({'pred':preds, 'obs': obs}))

print("\nPredicting on training and test set....")
training_predictions = predictions(X_train, y_train)
training_accuracy = sum(training_predictions.obs == training_predictions.pred) / float(training_predictions.shape[0])

test_predictions = predictions(X_test, y_test)
test_accuracy = sum(test_predictions.obs == test_predictions.pred) / float(test_predictions.shape[0])

print("Training Accuracy: " + str(training_accuracy))
print("Test Accuracy: " + str(test_accuracy))
"""
