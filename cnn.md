

# CS 168 Computational Methods for Medical Imaging 

**Automatic Lung's Grade Image Classification from CT**
by Reinaldo Daniswara and Tianyang Zhang

## 3D CNN
We are using tensorflow to transform and build a 3D-CNN model from 3D CT scan to create images classifier, which will be used to train the dataset that we have. 

We pick this method because unlike the traditional detection for the appearance of nodule in lung, 3D convolutional neural networks can learn useful representations directly from a large dataset without the need of extensive preprocessing pipelines. 

The basic architecture for our neural network is CT images -> convolution & pooling -> flatten layer -> fully connected layer -> output layer

In our implementation, we do convolutions and immediately followed by max-pooling method for 3 times, which are 32, 64, and 128 features. After that, before the data can be used as an input for a fully-connected layer, we need to convert the convolution result into a 2D tensor. At the end, we need to connect all the nodes before producing the output, and this is the duty of the fully connected layer become in handy.

In order to write this code, I look at the example tutorial for tensorflow from github user [aymericdamien](https://github.com/aymericdamien/TensorFlow-Examples)

```python
def con_3d(tensor, conv_outputs, kernel_size, stride): 
	#build conv function   
    weight = tf.Variable(tf.truncated_normal([kernel_size,
                                              kernel_size,
                                              kernel_size,
                                              tensor.get_shape().as_list()[4],
                                              conv_outputs],
                                              stddev=0.1))
    bias = tf.Variable(tf.zeros([conv_outputs]))
    
    tensor = tf.nn.con_3d(tensor,
                          weight,
                          strides = [1,stride,stride,stride,1],
                          padding = 'SAME')
    tensor = tf.nn.bias_add(tensor, bias)
   
    tensor = tf.nn.relu(tensor)
    
    return tensor

def maxpool(tensor, kernel_size, stride):    
    # max-pooling layer
    tensor = tf.nn.max_pool3d(tensor,
                              kernel_size = [1,kernel_size,kernel_size,kernel_size,1],
                              strides = [1,stride,stride,stride,1],
                              padding = 'SAME')
    
    return tensor

def flatten_layer(tensor):
	# flatten layer
    flattened = np.prod(tensor.get_shape().as_list()[1:])
    return tf.reshape(tensor, [-1, flattened])


def fully_conn(tensor, conv_outputs):  
	#fully connected Layer  
    size = tensor.get_shape().as_list()[1]
    weight = tf.Variable(tf.truncated_normal([size, conv_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros([conv_outputs]))
    tensor = tf.matmul(tensor, weight)
    tensor = tf.nn.bias_add(tensor, bias)
    tensor = tf.nn.relu(tensor)
    return tensor

def output(tensor, conv_outputs):
    size = tensor.get_shape().as_list()[1]
    weight = tf.Variable(tf.truncated_normal([size, conv_outputs], stddev=0.1))
    bias = tf.Variable(tf.zeros([conv_outputs]))
    
    # output layer
    tensor = tf.matmul(tensor, weight)
    tensor = tf.nn.bias_add(tensor, bias)
    
    return tensor

 def conv_net(x, keep_prob):
    # x to hold the input
    # keep_prob: placeholder tensor for the dropout layer
    
    # apply the convolutional and max-pooling layers
    x = maxpool(x, 2, 2) # applying a max-pooling layer first to reduce the memory usage
    
    con_1 = con_3d(x, 32, 5, 2)
    con_1 = maxpool(con_1, 2, 2)
    
    con_2 = con_3d(con_1, 64, 5, 2)
    con_2 = maxpool(con_2, 2, 2)

    con_3 = con_3d(con_2, 128, 5, 2)
    con_3 = maxpool(con_3, 2, 2)
    
    # flatten layer
    flat = flatten_layer(con_3)
    
    # fully-connected layers
    f_con_1 = fully_conn(flat, 1024)
    f_con_2 = fully_conn(f_con_1, 256)
    f_con_3 = fully_conn(f_con_2, 64)
    
    # dropout layer
    dropout = tf.nn.dropout(f_con_3, keep_prob)
    
    # output layer
    y = output(dropout, 2)
    
    return y

```
Tho build the network model, [here](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/4_Utils/save_restore_model.py) is there example code that I reference to.
The author write a really good example on how to develop a neural network model that we can follow easily.

Below is the code to build a neural network

```python
tf.reset_default_graph()

# inputs
x = tf.placeholder(tf.float32, (None, data_shape[0], data_shape[1], data_shape[2], 1), name='x')
y = tf.placeholder(tf.float32, (None, 2), name='y')

pl_prob = tf.placeholder(tf.float32, name='pl_prob')
logits = conv_net(x, pl_prob)
logits = tf.identity(logits, name='logits') # name logits tensor

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y), name='cost')
optimizer = tf.train.AdamOptimizer().minimize(cost) # Adam gradient descent optimizer

# accuracy
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
```
To train the model, this is also another reference that we follow:
https://min-sheng.github.io/deep-learning-cv/week3/Project/final_dlnd_image_classification_with_DA.html

```python
def train_cnn(session, optimizer, keep_probability, features_batch, labels_batch):
    session.run(optimizer, feed_dict={
        x: features_batch,
        y: labels_batch,
        pl_prob: keep_probability
    })

def stat_val(session, function, n_batch, set):
    if set == 'valid':
        assert (n_batch <= valid_batches.shape[0])
        batches = valid_batches
    if set == 'train':
        assert (n_batch <= train_batches.shape[0])
        batches = train_batches
    
    batch_indices = np.arange(batches.shape[0])
    shuffle(batch_indices) 
    # select first n_batch batches
    batch_indices = batch_indices[:n_batch] 
    
    total_size = 0.
    total_stats = 0.

    # Find the selected set
    for batch_index in batch_indices:
        features, labels = read_batch(batches[batch_index])
        current_stats = session.run(function, feed_dict={
                                    x: features,
                                    y: labels,
                                    pl_prob: 1. # deactivate dropout layer
                                    })
        current_batch_size = batch_size - np.sum(batches[batch_index] == '0')
        total_stats = total_stats + current_stats * current_batch_size
        total_size = total_size + current_batch_size
        stat_result = total_stats / total_size
    
    return stat_result
```

Printing the stat, tune up the hyperparameter, 
```python
def print_stat(session, cost, accuracy):
    # training loss
    loss = stat_val(session, cost, 200, 'train')
    # validation loss
    valid_loss = stat_val(session, cost, 50, 'valid')
    # validation accuracy
    valid_accuracy = stat_val(session, accuracy, 50, 'valid')
    
    print('Training Loss: {:>8.4f} Validation Loss: {:>8.4f} Validation Accuracy: {:.6f}'.format(
        loss, valid_loss, valid_accuracy))

# hyperparameters, can be tuned
epochs = 4
keep_probability = 0.7
model_save = '/Users/reinaldodaniswara/Desktop/cs168/project/code/patient_folder/data_folder/model1'

# start training
with tf.Session() as sess:
    # initialization
    sess.run(tf.global_variables_initializer())
    
    # saver
    saver = tf.train.Saver()
    
    # training
    for epoch in range(epochs):
        for batch_index in range(train_batches.shape[0]):
            features_batch, labels_batch = read_batch(train_batches[batch_index])
            train_cnn(sess, optimizer, keep_probability, features_batch, labels_batch)
            print('Epoch {:>2}, Batch {:>3} '.format(epoch + 1, batch_index), end='\r')
        print('Epoch {:>2} '.format(epoch + 1), end='')
        print_stat(sess, cost, accuracy)
        # save model
        saver.save(sess, model_save + '_' + str(epoch))
```
## Testing the data
```python
preimg_dir = '/Users/reinaldodaniswara/Desktop/cs168/project/code/patient_folder/data_folder/stage1_preprocessed/' 
data_shape = [250, 350, 350]
batch_size = 10 #to prevent running out of memory

patients = os.listdir(preimg_dir)
patients.sort()

test_size = 0

test_batch = []
curr_test_batch = []


for i in range(len(patients)):  
    if patients[i].startswith('.'): continue 
    
    if np.load(preimg_dir + patients[i])['set'] == 'test':
        curr_test_batch.append(patients[i])
        
        #  full batch handler
        if (len(curr_test_batch) == batch_size): 
            test_batch.append(curr_test_batch)
            test_size = test_size + batch_size
            curr_test_batch = []

# save the remaining test data
if len(curr_test_batch) != 0:
    test_size = test_size + len(curr_test_batch)
    # pad zeros to make its size equal to batch_size
    while (len(curr_test_batch) != batch_size):
        curr_test_batch.append(0)
    test_batch.append(curr_test_batch)
    curr_test_batch = []

test_batch = np.array(test_batch)

def batch_in(batch_files):
    current_batch_size = batch_size - np.sum(batch_files == '0')
    
    batch_features = np.zeros((current_batch_size, data_shape[0], data_shape[1], data_shape[2], 1))
    batch_ids = []
    
    for i in range(len(batch_files)):
        if batch_files[i] != '0':
            data = np.load(preimg_dir + batch_files[i])
            batch_features[i,:,:,:,0] = data['data']
            batch_ids.append(batch_files[i][:32])

    return batch_features, batch_ids

test_features_batch_sample, test_ids_batch_sample = batch_in(test_batch[0])

#plt.imshow(test_features_batch_sample[0,:,:,100,0], cmap=plt.cm.bone)
#plt.show()

#--- test model ---- 
model_save = '/Users/reinaldodaniswara/Desktop/cs168/project/code/patient_folder/data_folder/model1'
result_save = '/Users/reinaldodaniswara/Desktop/cs168/project/code/patient_folder/data_folder/cancer.csv'

cnn_graph = tf.Graph()
open_file = open(result_save, 'w')
open_file.write('id,cancer\n') # header line

with tf.Session(graph=cnn_graph) as sess:
    loader = tf.train.import_meta_graph(model_save + '.meta')
    loader.restore(sess, model_save)
    
    # get tensors
    x = cnn_graph.get_tensor_by_name('x:0')
    y = cnn_graph.get_tensor_by_name('y:0')
    pl_prob = cnn_graph.get_tensor_by_name('pl_prob:0')
    logits = cnn_graph.get_tensor_by_name('logits:0')

    for batch_index in range(test_batch.shape[0]):
        test_features_batch, test_ids_batch = batch_in(test_batch[batch_index])
        predictions = sess.run(tf.nn.softmax(logits), feed_dict={
            x: test_features_batch,
            pl_prob: 1.
        })
        
        for test_index in range(len(test_ids_batch)):
            open_file.write(test_ids_batch[test_index] + ',' + str(predictions[test_index,1]) + '\n')
            print('ID: ' + test_ids_batch[test_index] + ', Cancer probability: ' + str(predictions[test_index,1]))

open_file.close()
```
[<< Go to preprocessing code](http://reinaldodaniswara.com/preprocessing.html)

[<< Go back to the weekly progress page](http://reinaldodaniswara.com/medicalimaging.html)
