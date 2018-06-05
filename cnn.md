
# CS 168 Computational Methods for Medical Imaging 

**Automatic Lung's Grade Image Classification from CT**
by Reinaldo Daniswara and Tianyang Zhang

## 3D CNN
We are using tensorflow to transform and build a 3D-CNN model from 3D CT scan to create images classifier, which will be used to train the dataset that we have. 

We pick this method because unlike the traditional detection for the appearance of nodule in lung, 3D convolutional neural networks can learn useful representations directly from a large dataset without the need of extensive preprocessing pipelines. 

The basic architecture for our neural network is CT images -> convolution & pooling -> flatten layer -> fully connected layer -> output layer

In our implementation, we do convolutions and immediately followed by max-pooling method for 3 times, which are 32, 64, and 128 features. After that, before the data can be used as an input for a fully-connected layer, we need to convert the convolution result into a 2D tensor. At the end, we need to connect all the nodes before producing the output, and this is the duty of the fully connected layer become in handy.

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
