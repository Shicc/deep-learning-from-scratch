import pandas as pd
import numpy as np
import tensorflow as tf

# 参数，超参数
batch_size = 50
lr = 0.1

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
x = tf_input[:,:1600]
w1 = tf.Variable(tf.zeros([1600,100]))
b1 = tf.Variable(tf.zeros([100]))
h1 = tf.matmul(x,w1)+b1
w2 = tf.Variable(tf.zeros([100,20]))
b2 = tf.Variable(tf.zeros([20]))
h2 = tf.matmul(h1,w2)+b2
w3 = tf.Variable(tf.zeros([20,2]))
b3 = tf.Variable(tf.zeros([2]))
pre_y = tf.nn.softmax(tf.matmul(h2,w3)+b3)

y = tf_input[:,1600:]            # onehot

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pre_y,labels = y))
train_op = tf.train.AdamOptimizer(lr).minimize(loss)

accuracy = tf.metrics.accuracy(
        labels = tf.argmax(y,axis = 1),
        predictions = tf.argmax(pre_y,axis = 1)
)[1]

sess = tf.Session()
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
        predictions_2 = tf.argmax(pre_y,axis = 1)
        test_data_ = test_data[:50]
        acc_,predictions_2_ ,output_= sess.run([accuracy,predictions_2,pre_y], feed_dict = {tf_input: test_data_})
        print('predictions_2:', predictions_2_,'output:',output_)
        print('accuracy:',acc_)
