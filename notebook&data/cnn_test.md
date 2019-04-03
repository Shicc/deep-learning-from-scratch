

```python
import pandas as pd
import tensorflow as tf
# import numpy as np

# tf.set_random_seed(1)
# np.random.seed(1)

# 参数，超参数
batch_size = 50
lr = 0.5

def to_onehot(y):
    for i in range(len(y)):
        if y.iloc[i,0]==0:
            y.iloc[i,0] = 'y'
        else:
            y.iloc[i,0] = 'f'
    return pd.get_dummies(y,prefix=y.columns)

# 准备数据
train_data_set = pd.read_csv('train.csv')                   # shape:(4000,1602)
train_data = train_data_set.iloc[0:4000,1:1601]       # shape:(4000,1600)
train_labels = to_onehot(train_data_set.iloc[0:4000,[1601]])           # shape:(4000,1)
print('----')
```

    ----
    


```python
# 先取一小批测试数据看看结果
test_data_set = pd.read_csv('test.csv')                 # shape:(3550,1601) 没有包含正确结果，和训练集不一样
test_data = test_data_set.iloc[0:1000,1:1601]           # shape:(1000,1600)
test_labels = to_onehot(
    pd.read_csv('sample_submit.csv').iloc[0:1000,[1]])          # shape:(1000,1) 一一对应
print('----')
```

    ----
    


```python
# 创建模型
tf_x = tf.placeholder(tf.float32,[None,40*40])          # 每批50个数据，此处暂时不管，为None
x_img = tf.reshape(tf_x,[-1,40,40,1])                   # (batch, height, width, channel)
tf_y = tf.placeholder(tf.float32,[None,2])

## CNN
conv1 = tf.layers.conv2d(           # shape:(40,40,8)
        inputs = x_img,
        filters = 10,               # 图像卷积后的深度
        kernel_size = 5,            # 扫描核5*5大小
        strides = 1,
        padding = 'same',
        activity_regularizer = tf.nn.relu
)                                   # shape:(40,40,10)
pool1 = tf.layers.max_pooling2d(
        inputs = conv1,
        pool_size = 2,
        strides = 2
)                                   # shape:(20,20,10)
conv2 = tf.layers.conv2d(
        inputs = pool1,
        filters = 20,
        kernel_size = 5,
        strides = 1,
        padding = 'same',
        activity_regularizer = tf.nn.relu
)                                   # shape:(20,20,20)
pool2 = tf.layers.max_pooling2d(
        inputs = conv2,
        pool_size = 2,
        strides = 2
)                                   # shape:(10,10,20)
flat_data = tf.reshape(pool2,[-1,10*10*20]) #(10*10*20, )
output = tf.layers.dense(flat_data,2) #用于全连接层，一共输出两种分类个数

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits = output)
# loss = tf.reduce_mean(-tf.reduce_sum(tf_y * tf.log(output),reduction_indices=[1]))
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)
print('----')
```

    WARNING:tensorflow:From d:\program files\python\lib\site-packages\tensorflow\python\ops\losses\losses_impl.py:691: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    
    Future major versions of TensorFlow will allow gradients to flow
    into the labels input on backprop by default.
    
    See tf.nn.softmax_cross_entropy_with_logits_v2.
    
    ----
    


```python
# 计算精度
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
print('----')
```

    ----
    


```python
for step in range(1001):
    b_x = train_data.iloc[step*batch_size:(step+1)*batch_size]
    b_y = train_labels.iloc[step*batch_size:(step+1)*batch_size]
    # _,loss_ = sess.run([train_op,loss],feed_dict = {tf_x:b_x,tf_y:b_y})
    sess.run(train_op,feed_dict = {tf_x:b_x,tf_y:b_y})
    if step % 200 ==0:
        accuracy_= sess.run(accuracy,feed_dict = {tf_x:test_data,tf_y:test_labels})
        print('Step:', step,'| test accuracy: %.8f' % accuracy_)
    if step == 1000:
        predictions_2 = tf.argmax(output,axis = 1)
        test_data_ = test_data.iloc[23:167]
        predictions_2_ ,output_= sess.run([predictions_2,output], feed_dict = {tf_x: test_data_})
        print('predictions_2:', predictions_2_,'output:',output_)
        
#当第一次运行完了后，就可以注释掉保存的网络了。
# 改参数可以在这里改，然后保存的模型位置又需要换一个，以免覆盖以前的。
#   保存网络
# saver = tf.train.Saver()
# save_path = saver.save(sess,"my_net/cnn_net.ckpt",write_meta_graph = False)
# print("save_path:",save_path) #打印出来看看
```

    Step: 0 | test accuracy: 0.50000000
    Step: 200 | test accuracy: 0.50000000
    Step: 400 | test accuracy: 0.50000000
    Step: 600 | test accuracy: 0.50000000
    Step: 800 | test accuracy: 0.50000000
    Step: 1000 | test accuracy: 0.50000000
    predictions_2: [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] output: [[ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]
     [ nan  nan]]
    


```python
b_y = test_labels.iloc[0:batch_size]
print(b_y)
# print(b_y.columns)
b_y.to_csv("b_y.csv",
          index=False,header=False
          )
# b_y_onehot = pd.get_dummies(b_y,prefix=b_y.columns)
# print(b_y_onehot)
# print('b_y[0,0]',train_labels.iloc[0,0])
```

        y
    0   1
    1   0
    2   1
    3   0
    4   1
    5   0
    6   1
    7   0
    8   1
    9   0
    10  1
    11  0
    12  1
    13  0
    14  1
    15  0
    16  1
    17  0
    18  1
    19  0
    20  1
    21  0
    22  1
    23  0
    24  1
    25  0
    26  1
    27  0
    28  1
    29  0
    30  1
    31  0
    32  1
    33  0
    34  1
    35  0
    36  1
    37  0
    38  1
    39  0
    40  1
    41  0
    42  1
    43  0
    44  1
    45  0
    46  1
    47  0
    48  1
    49  0
    


```python
by = pd.read_csv('b_y.csv',names=["y_label"])
print(by)
print(by.columns)
```

        y_label
    0         1
    1         0
    2         1
    3         0
    4         1
    5         0
    6         1
    7         0
    8         1
    9         0
    10        1
    11        0
    12        1
    13        0
    14        1
    15        0
    16        1
    17        0
    18        1
    19        0
    20        1
    21        0
    22        1
    23        0
    24        1
    25        0
    26        1
    27        0
    28        1
    29        0
    30        1
    31        0
    32        1
    33        0
    34        1
    35        0
    36        1
    37        0
    38        1
    39        0
    40        1
    41        0
    42        1
    43        0
    44        1
    45        0
    46        1
    47        0
    48        1
    49        0
    Index(['y_label'], dtype='object')
    


```python
print(by.keys())
for name in by.keys():
    print(name,pd.unique(by[name]))
```

    Index(['y_label'], dtype='object')
    y_label [1 0]
    


```python
by_one_hot = pd.get_dummies(by)
print(by_one_hot)
```

        y_label
    0         1
    1         0
    2         1
    3         0
    4         1
    5         0
    6         1
    7         0
    8         1
    9         0
    10        1
    11        0
    12        1
    13        0
    14        1
    15        0
    16        1
    17        0
    18        1
    19        0
    20        1
    21        0
    22        1
    23        0
    24        1
    25        0
    26        1
    27        0
    28        1
    29        0
    30        1
    31        0
    32        1
    33        0
    34        1
    35        0
    36        1
    37        0
    38        1
    39        0
    40        1
    41        0
    42        1
    43        0
    44        1
    45        0
    46        1
    47        0
    48        1
    49        0
    


```python
df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [1, 2, 3]})
print(df)
pd.get_dummies(df, prefix=['col1', 'col2'])
```

       A  B  C
    0  a  b  1
    1  b  a  2
    2  a  c  3
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>C</th>
      <th>col1_a</th>
      <th>col1_b</th>
      <th>col2_a</th>
      <th>col2_b</th>
      <th>col2_c</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
b_x = train_data.iloc[0:batch_size]
print(b_x)  #老是有标签id,p_i_j等，不知输出的时候会不会带上而导致的出错。
```

        p_0_0  p_0_1  p_0_2  p_0_3  p_0_4  p_0_5  p_0_6  p_0_7  p_0_8  p_0_9  \
    0     176    224    164     66    255     79    255    156    118    196   
    1     201    113    150    201    132     22    215     99     10    231   
    2     255    127    186    255    125    107     36    227     54    125   
    3     255    214    179     53    231    242    248     11    146     87   
    4     158    134    249    174    140     23     98    194    255    137   
    5     255    133    210    220    138      1    172    246     90     43   
    6     210     43    146     12    138     26    141    195      7    133   
    7      99     67    229     79    255    133    167    255    222    248   
    8      44    250    148     10     15    163    206    150    199     10   
    9     135    255    224     27    202    129    255    165    171    252   
    10     75    177    173    203    187     97    228    113    150    123   
    11     86    202    138    194    174    208    255     43     29    155   
    12     90    148    112    147    255    129     58    254     82    255   
    13    132     76     70    125     16    147     17    255     80    255   
    14     32    134    255    157     26    130     58    169     56    190   
    15    199    238    139    172    166    255    135    191     41    134   
    16    198    216    157    169    204    205    161    187     17    100   
    17    145    113     78    235     61    123    222     72    107    254   
    18     57     76     59    194    103     54    255    178     43    126   
    19    138    255     71    171     26     51    243    179    149    255   
    20    174    240    147    107     89     31    255     96    255     46   
    21     79    211     98    255    214     70    210      9    194    153   
    22    107    255    179     16    139    255    113    201     68     27   
    23     68      6     54    255    237    147    137     49    234     49   
    24    246    200    255    169    112    232     75     59      1     75   
    25    207    252     60     73    255    137     16    116    151    212   
    26    255    119    162    109    255     24    127     47     27     32   
    27    205     28    109    114    255    114      7     83    245    208   
    28    132     66     26    111    243    205    167     49     57    255   
    29     13    137    255     64    255     83    150    127    184     80   
    30    146     35    255    215     32    153     74    120    139    119   
    31    140     88     20    183    251    255     59    255     67    185   
    32     69    174    157      1    184    171    165    241    180    201   
    33     82    255    164    165    247     88    245    142    243     38   
    34     48    184    220    190    255    150    186    239    255    217   
    35    226    228    140     70    198     58    200    188    225      0   
    36    255     42    200     37    164     12     27    255     44     41   
    37    255    249     51    215     71    232     62     75    172    255   
    38    255    221    214    255    129    255     90    198     46    105   
    39     23    161     55     76     14    140     90    161     37     55   
    40     84    200    216    247     64     57    173    191    180    165   
    41     72    255     35    183    143     40      4    252     34    255   
    42      9    143     25    199    177    147    100     99    211    173   
    43     14     95    126    253    207    214    213    255    242    255   
    44    122     41     23    175     61    239    162    210     58    164   
    45     89    213    158     26    148    108    252    216    181    255   
    46    138     93    197     95      5    170    255    255    170    255   
    47    218    213     19     94     12    148    177    234    147    226   
    48    151    244    105    170    120     83    118    221    114     20   
    49    138    105    104    147     15    100    188     63     13    180   
    
         ...     p_39_30  p_39_31  p_39_32  p_39_33  p_39_34  p_39_35  p_39_36  \
    0    ...          42       16      192      244      171      193      250   
    1    ...         174      190       55       65       59       57       18   
    2    ...           7       94      119      138      255      255      175   
    3    ...         187      255      167      128        5       68      171   
    4    ...          59      255       23       98       32       35       40   
    5    ...          15      167      108      143        8       64      113   
    6    ...         113      232      255      255      143      161      184   
    7    ...         107      190      105       23      145       12      162   
    8    ...          86      255       56      242      197      186      235   
    9    ...         160      254      219      219       39      238       30   
    10   ...         231       84       61       86      255      120      100   
    11   ...          21       19      163       29       73       51       11   
    12   ...          56       83      255       45      255      248       36   
    13   ...         200        6      115       29      230      178      212   
    14   ...           8      231      232      168       89      188      250   
    15   ...         255       84       51       58      135      188       25   
    16   ...         224        4      218      255       64      255       63   
    17   ...         168       27      102       66      179      225        2   
    18   ...         255      211       45      156      147      165        7   
    19   ...         255      140       37      183       98       68        3   
    20   ...         177      106       24       85       97      255      255   
    21   ...         177       86      143      123      171      208      101   
    22   ...         255       30      192        3      153       97      224   
    23   ...         221       92      200       25      194      194      227   
    24   ...         174        6      245      189      181      154       62   
    25   ...          74       21      145       85       35        9      118   
    26   ...          42        8      200      111      158       93      226   
    27   ...         233       97      237      125      121       59      143   
    28   ...         112      102        8      184      212      233      179   
    29   ...         252       74      228       21      140       48      119   
    30   ...         189      185      227      211      209      255       44   
    31   ...         153      255      135       68       51      180       34   
    32   ...         255       71      161      255       73      236      194   
    33   ...         255       34       11       29      231       41       52   
    34   ...         255       62        7      222      255      185       70   
    35   ...         255      127       84      227      224      196      173   
    36   ...         255       95       56      241      180      208      255   
    37   ...         217      202       24      143      255      255      255   
    38   ...          35      255       48      255      122       67      112   
    39   ...         243      117       57       77      255      255       21   
    40   ...         157      136       66       66      126      212      106   
    41   ...         205       46      197       93       77       55       72   
    42   ...          92      234      220      210      217      123      145   
    43   ...         170       33      177      176      233       73      140   
    44   ...           5       25      153       31      132       99      194   
    45   ...          75      166       72      255      236       29      190   
    46   ...          49      251       58       40      156      208      100   
    47   ...          43      156      246      113      174      119      153   
    48   ...         255       43      187      146       83      146      250   
    49   ...         255      178       67      221       11      250      111   
    
        p_39_37  p_39_38  p_39_39  
    0       247      240      237  
    1       131      199      161  
    2       255       49       98  
    3       120      141       16  
    4        35      124      169  
    5       114      216      131  
    6        41      194       22  
    7       255      139       48  
    8       138      255      102  
    9       204      213       49  
    10       66       45       79  
    11      255       88      162  
    12      230      207       13  
    13      213      197      255  
    14       11       76      167  
    15      175      123      198  
    16       90       41      157  
    17       25      220      239  
    18      189      146      255  
    19       25      201       49  
    20      255      138       70  
    21      143      104      239  
    22      255      129      255  
    23      103      201       64  
    24       37      102      183  
    25      173      255      255  
    26       72       60       66  
    27       14      255      217  
    28      126      106       93  
    29      174       81      207  
    30       57      184       77  
    31      122      239      218  
    32       29      174      255  
    33        2      169      155  
    34      115       78       69  
    35      230        8      255  
    36      145      233      255  
    37      255      120      255  
    38      212       94      138  
    39      231      110       68  
    40        9      254      138  
    41        7      221      210  
    42       55       93      122  
    43      185       31      222  
    44      226      255      221  
    45      136      117      102  
    46      125       25      187  
    47      140      227       30  
    48       34       68      255  
    49       35      211      247  
    
    [50 rows x 1600 columns]
    


```python
tf_x_ = tf.placeholder(tf.int32,[None,40*40]) 
x_img_ = tf.reshape(tf_x_,[-1,40,40,1])
print(sess.run(x_img_,feed_dict={tf_x_:b_x}))
```

    [[[[176]
       [224]
       [164]
       ..., 
       [228]
       [225]
       [222]]
    
      [[ 27]
       [241]
       [ 19]
       ..., 
       [ 43]
       [255]
       [145]]
    
      [[193]
       [142]
       [ 74]
       ..., 
       [182]
       [213]
       [ 26]]
    
      ..., 
      [[218]
       [255]
       [140]
       ..., 
       [166]
       [242]
       [178]]
    
      [[101]
       [  4]
       [211]
       ..., 
       [234]
       [ 49]
       [ 65]]
    
      [[186]
       [ 25]
       [107]
       ..., 
       [247]
       [240]
       [237]]]
    
    
     [[[201]
       [113]
       [150]
       ..., 
       [ 69]
       [117]
       [247]]
    
      [[187]
       [255]
       [232]
       ..., 
       [ 47]
       [205]
       [ 76]]
    
      [[248]
       [  7]
       [ 95]
       ..., 
       [228]
       [192]
       [ 29]]
    
      ..., 
      [[ 93]
       [127]
       [ 33]
       ..., 
       [255]
       [ 40]
       [230]]
    
      [[255]
       [ 63]
       [ 35]
       ..., 
       [ 10]
       [137]
       [  0]]
    
      [[118]
       [207]
       [170]
       ..., 
       [131]
       [199]
       [161]]]
    
    
     [[[255]
       [127]
       [186]
       ..., 
       [ 48]
       [ 22]
       [139]]
    
      [[255]
       [133]
       [ 16]
       ..., 
       [ 30]
       [ 20]
       [ 69]]
    
      [[ 42]
       [ 96]
       [ 46]
       ..., 
       [255]
       [ 64]
       [251]]
    
      ..., 
      [[131]
       [255]
       [254]
       ..., 
       [229]
       [ 24]
       [218]]
    
      [[146]
       [210]
       [135]
       ..., 
       [224]
       [204]
       [166]]
    
      [[ 40]
       [203]
       [255]
       ..., 
       [255]
       [ 49]
       [ 98]]]
    
    
     ..., 
     [[[218]
       [213]
       [ 19]
       ..., 
       [255]
       [220]
       [145]]
    
      [[154]
       [221]
       [204]
       ..., 
       [252]
       [  3]
       [190]]
    
      [[ 69]
       [ 57]
       [114]
       ..., 
       [229]
       [155]
       [251]]
    
      ..., 
      [[134]
       [ 11]
       [ 73]
       ..., 
       [ 58]
       [147]
       [127]]
    
      [[255]
       [106]
       [210]
       ..., 
       [158]
       [201]
       [255]]
    
      [[ 20]
       [255]
       [178]
       ..., 
       [140]
       [227]
       [ 30]]]
    
    
     [[[151]
       [244]
       [105]
       ..., 
       [128]
       [148]
       [255]]
    
      [[137]
       [133]
       [225]
       ..., 
       [169]
       [210]
       [197]]
    
      [[226]
       [101]
       [ 42]
       ..., 
       [ 26]
       [ 18]
       [255]]
    
      ..., 
      [[ 42]
       [ 34]
       [  4]
       ..., 
       [  7]
       [  6]
       [ 39]]
    
      [[255]
       [161]
       [ 25]
       ..., 
       [248]
       [141]
       [249]]
    
      [[145]
       [140]
       [182]
       ..., 
       [ 34]
       [ 68]
       [255]]]
    
    
     [[[138]
       [105]
       [104]
       ..., 
       [255]
       [  2]
       [ 42]]
    
      [[ 34]
       [ 57]
       [236]
       ..., 
       [177]
       [210]
       [255]]
    
      [[  7]
       [ 15]
       [146]
       ..., 
       [255]
       [121]
       [171]]
    
      ..., 
      [[111]
       [ 49]
       [121]
       ..., 
       [255]
       [ 69]
       [178]]
    
      [[115]
       [255]
       [162]
       ..., 
       [241]
       [167]
       [204]]
    
      [[185]
       [145]
       [138]
       ..., 
       [ 35]
       [211]
       [247]]]]
    


```python
tf_y = tf.placeholder(tf.float32,[None,2])
y = tf.argmax(tf_y,axis = 1)
with tf.Session() as sess1:
    sess1.run(tf.global_variables_initializer())
    print(sess1.run(tf_y,feed_dict={tf_y:train_labels.iloc[12:40]}))
    print(sess1.run(tf.argmax(tf_y,axis = 1),feed_dict={tf_y:train_labels.iloc[12:40]}))
    

```

    [[ 1.  0.]
     [ 0.  1.]
     [ 1.  0.]
     [ 1.  0.]
     [ 0.  1.]
     [ 0.  1.]
     [ 1.  0.]
     [ 1.  0.]
     [ 1.  0.]
     [ 0.  1.]
     [ 0.  1.]
     [ 1.  0.]
     [ 1.  0.]
     [ 0.  1.]
     [ 0.  1.]
     [ 1.  0.]
     [ 0.  1.]
     [ 1.  0.]
     [ 1.  0.]
     [ 0.  1.]
     [ 0.  1.]
     [ 0.  1.]
     [ 1.  0.]
     [ 0.  1.]
     [ 0.  1.]
     [ 1.  0.]
     [ 0.  1.]
     [ 1.  0.]]
    [0 1 0 0 1 1 0 0 0 1 1 0 0 1 1 0 1 0 0 1 1 1 0 1 1 0 1 0]
    
