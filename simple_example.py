# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 20:38:03 2018
首先来生成输入数据。我们假设最后要学习的方程为y = x2 − 0.5，我们来构造满足这个方
程的一堆x 和y，同时加入一些不满足方程的噪声点
@author: Fuxiao
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


'''构建网络模型'''
def add_layer(inputs, in_size, out_size, activation_function=None):
    weights=tf.Variable(tf.random_normal([in_size, out_size]))
    bias=tf.Variable(tf.zeros([1, out_size])+0.1) #偏置也应该定义为变量Variable的形式
    Wx_plus_b=tf.matmul(inputs, weights)+bias #注意这里的输入与权重矩阵乘法，顺序不能乱，否则会出错
    
    if activation_function==None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs


## 构造满足一元二次方程的函数
x_data=np.linspace(-1, 1, 300)[:, np.newaxis]
'''采用np 生成等差数列的方法，并将结果为300 个点的一维数组，转换为300×1 的二维数组,
原来是300个数，现在变成了(300,1)的数组
'''
noise = np.random.normal(0, 0.05, x_data.shape) # 加入一些噪声点，使它与x_data 的维度一致，并且拟合为均值为0、方差为0.05 的正态分布
y_data=np.square(x_data)-0.5+noise #y=x^2-0.5+噪声

#定义x,y的占位符，一般来说x,y之类的就用占位符，而参数就写成变量Variable
xs=tf.placeholder(tf.float32, [None, 1])
ys=tf.placeholder(tf.float32, [None, 1])


#构建输入层到隐藏层的神经网络
h1=add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction=add_layer(h1, 10, 1, activation_function=None)
    
#构建损失函数
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction), reduction_indices=[1]))
#最小化损失函数, 梯度下降法
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
  
#整个网络定义好了以后开始初始化运行
init=tf.global_variables_initializer()#s所有的变量初始化
with tf.Session() as sess:
    sess.run(init)
    # plot the real data
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x_data, y_data)
    plt.ion()
    
    for i in range(5000):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if i % 50 == 0: # 每50 次打印出一次损失值
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
    # plot the prediction
    lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
    plt.pause(0.9)




























