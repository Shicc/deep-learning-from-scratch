import pandas as pd
import numpy as np
import tensorflow as tf



# 参数，超参数
batch_size = 50
lr = 0.5

def to_onehot(y):
    for i in range(len(y)):
        if y.iloc[i,1600]==0:
            y.iloc[i,1600] = 'y'
        else:
            y.iloc[i,1600] = 'f'
    return pd.get_dummies(y)

# 准备数据
data_set = pd.read_csv('train.csv').iloc[:,1:]
# 仅把标签转化成onehot
data_set = to_onehot(data_set)

# 打乱数据，使数据随机
data_set = data_set.values.astype(np.float32)
np.random.shuffle(data_set)

# 划分训练集和验证集
sep = int(0.7*len(data_set))
train_data = data_set[:sep]
test_data = data_set[sep:]

# 创建模型
tf_input = tf.placeholder(tf.float32,[None,1602])
tf_x = tf_input[:,:1600]
x_img = tf.reshape(tf_x,[-1,40,40,1])
tf_y = tf_input[:,1600:]            # onehot

## CNN
conv1 = tf.layers.conv2d(           # shape:(40,40,1)
        inputs = x_img,
        filters = 10,               # 图像卷积后的深度
        kernel_size = 5,            # 扫描核5*5大小
        padding = 'same',
        activity_regularizer = tf.nn.relu
)                                   # shape:(40,40,10)
pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = [2,2],
        strides = 2
)                                   # shape:(20,20,10)
conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 20,
        kernel_size = 5,
        padding = 'same',
        activity_regularizer = tf.nn.relu
)                                   # shape:(20,20,20)
pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = [2,2],
        strides = 2
)                                   # shape:(10,10,20)

shape = pool2.get_shape().as_list() # (ง •_•)ง
flat_data = tf.reshape(pool2,[-1,shape[1]*shape[2]*shape[3]])
output = tf.layers.dense(flat_data,2)
loss = tf.losses.softmax_cross_entropy(onehot_labels = tf_y,logits = output)
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

# 计算精度
accuracy = tf.metrics.accuracy(
        labels = tf.argmax(tf_y,axis = 1),
        predictions = tf.argmax(output,axis = 1)
)[1]

# 重要步骤！！初始化
sess = tf.Session()
#初始化全局和本地变量
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# 训练
for step in range(1201):
    batch_index = np.random.randint(len(train_data), size = batch_size)
    sess.run(train_op,{tf_input:train_data[batch_index]})
    if step % 50 ==0:
        acc_, loss_ = sess.run([accuracy, loss], {tf_input: test_data})
        print("Step: %i" % step, "| Accurate: %.8f" % acc_, "| Loss: %.2f" % loss_ )

    if step == 1200:
        predictions_2 = tf.argmax(output,axis = 1)
        test_data_ = test_data[:100]
        acc_,predictions_2_ ,output_,tf_y_= sess.run([accuracy,predictions_2,output,tf_y], feed_dict = {tf_input: test_data_})
        print('predictions_2:', predictions_2_,'tf_y_',np.argmax(test_data_[:,1600:],axis = 1))
        print('accuracy:',acc_)
        print('output:',output_)







