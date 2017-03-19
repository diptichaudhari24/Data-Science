################################################################
################################################################
# libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler
import random
random.seed(4)
np.random.seed(4)

################################################################
################################################################
# read and process dataset
dataset = pd.read_csv("/home/sangram/Documents/Thesis/breast-cancer-wisconsin.csv")
dataset.columns = ['Id','Thickness','Uni_cell_Size', 'Uni_cell_Shape','Marginal_Adhesion','Epithelial','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses','Class']
dataset.replace('?', np.nan, inplace=True)
dataset = dataset.dropna()

# divide data into train and test sets
X_data = dataset[['Thickness','Uni_cell_Size', 'Uni_cell_Shape','Marginal_Adhesion','Epithelial','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses']]
y_data = dataset['Class']



sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
for train_index, test_index in sss.split(X=X_data, y=y_data):
  pass

X_train = np.array(X_data.iloc[train_index])
X_test  = np.array(X_data.iloc[test_index])

y_train = y_data.iloc[train_index]
y_test  = y_data.iloc[test_index]

# normalize X_train and X_test

# min_max_scaler = MinMaxScaler()
# min_max_scaler.fit(X_train)
# X_train = min_max_scaler.transform(X_train)
# X_test  = min_max_scaler.transform(X_test)

# convert y labels to one hot encoded vectors
# False: [0, 1] and True: [1, 0]
y_train = np.asarray([[1., 0.] if y_train.iloc[i]==2 else [0., 1.] for i in range(y_train.shape[0])])
y_test  = np.asarray([[1., 0.] if y_test.iloc[i]==2  else [0., 1.] for i in range(y_test.shape[0])])
################################################################
################################################################
# define tensorflow network

input_size   = 9
output_size  = 2
hidden_size1 = 10
hidden_size2 = 10

x_tensor = tf.placeholder(dtype=tf.float32, shape=[None, input_size])
y_tensor = tf.placeholder(dtype=tf.float32, shape=[None, output_size])

# h1 = sigmoid(Wx + b)
weights_input_to_hidden1 = tf.Variable(tf.truncated_normal(shape=[input_size, hidden_size1]))
biases_hidden1           = tf.Variable(0.1 * tf.ones(shape=[1, hidden_size1]))
hidden_layer1            = tf.nn.sigmoid(tf.add(tf.matmul(x_tensor, weights_input_to_hidden1), biases_hidden1))

# h2 = sigmoid(Wh1 + b)
weights_input_to_hidden2 = tf.Variable(tf.truncated_normal(shape=[hidden_size1, hidden_size2]))
biases_hidden2           = tf.Variable(0.1 * tf.ones(shape=[1, hidden_size2]))
hidden_layer2            = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer1, weights_input_to_hidden2), biases_hidden2))

# o = softmax(Wh2 + b)
weights_hidden_to_output = tf.Variable(tf.truncated_normal(shape=[hidden_size2, output_size]))
biases_output            = tf.Variable(0.1 * tf.ones(dtype= tf.float32, shape=[1,output_size]))
output_layer             = tf.nn.softmax(tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer2, weights_hidden_to_output), biases_output)))

# define training steps
loss          = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_tensor, logits=output_layer))
optimizer     = tf.train.AdadeltaOptimizer(learning_rate=0.7)
training_step = optimizer.minimize(loss)

################################################################
################################################################
# do the training!

sess = tf.Session()
sess.run(tf.global_variables_initializer())

n_epochs = 100
for epoch in range(n_epochs):
  loss_per_epoch = []
  for i in range(X_train.shape[0]):  
    x_instance = np.array([X_train[i]])
    y_instance = np.array([y_train[i]])
    _, loss_for_this_instance = sess.run([training_step, loss], feed_dict={x_tensor:x_instance, y_tensor:y_instance})
    loss_per_epoch.append(loss_for_this_instance)
  print ('Training Loss for ' + str(epoch+1) + ' epoch: ' + str(np.mean(loss_per_epoch)))
  
  # get test data loss after this epoch
  print ('Test Loss for ' + str(epoch+1) + ' epoch: ' + str(sess.run(loss, feed_dict={x_tensor:X_test, y_tensor:y_test})))
  print ('')

################################################################
################################################################
# evaluate on train/test data
train_predictions = np.argmax(sess.run(output_layer, feed_dict={x_tensor:X_train}), axis=1)
train_labels      = np.argmax(y_train, axis=1)

print ('Train Accuracy: ' + str(np.sum(np.equal(train_predictions, train_labels)) / float(y_train.shape[0])))

test_predictions = np.argmax(sess.run(output_layer, feed_dict={x_tensor:X_test}), axis=1)
test_labels      = np.argmax(y_test, axis=1)

print ('Test Accuracy: ' + str(np.sum(np.equal(test_predictions, test_labels)) / float(y_test.shape[0])))
