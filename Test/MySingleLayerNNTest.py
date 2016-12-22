#-*- coding: utf-8 -*-
# softmax 회귀 분석 예제
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("My_MNIST_data", one_hot=True) # hot 이 의미 하는건 여려 결과중 하나만 선정 하겠다는것

import tensorflow as tf

print(mnist.train.images)
print(tf.convert_to_tensor(mnist.train.images).get_shape())

#가설  784 => 28 * 28 : mnist image 의 가로 세로 크기 10 => 0 ~ 9 까지의 결과 의미
W = tf.Variable(tf.zeros([784, 10])) # 2차원 텐서
b = tf.Variable(tf.zeros([10]))

#입력, 출력 변수
x = tf.placeholder("float", [None, 784]) # 1차원 텐서
y = tf.nn.softmax(tf.matmul(x,W)+b)

#예측값
y_ = tf.placeholder("float", [None, 10]) # 1차원 텐

# 예측값(y_) 에 실제값 log(y) 를 곱하여 오차를 감지
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) # log 를 취한 이유가 뭘까?, 로그 함수 정리 참고 필요 http://pythonkim.tistory.com/28
# log(x) : x 가 0 일때 음수 무한,  1일때 0
# 한마디로 1에 가까울 수록 오차가 없고 그렇지 않을 경우에는 오차를 크게 만들어서 차이를 크게둠

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) ## cost 함수

sess = tf.Session();
sess.run(tf.global_variables_initializer())

for i in range(1000) :
    # xs : image, ys : label

    # next_batch 코드는 튜플을 실행중인 TensorFLow 세션에 넣기 위해 사용함
    batch_xs, batch_ys = mnist.train.next_batch(100) # 랜덤한 100개의 image, label 데이터 추출,

    #print('----------- xxxxxxx ')
    #print(batch_xs.get_shape())
    #print(batch_xs)
    #print('----------- yyyyyyyyyy')
    #print(batch_ys.get_shape())
    #print(batch_ys)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) # argmax 에 대한 설명 http://pythonkim.tistory.com/73
                                                                    # 2차원 배열[None, 10] 에서 2차원에 해당 하는 값중 가장 큰값을 가지고 오라는 것이고 그 값이 같은지 판별 하는곳
    #print('---- correct predict ')
    #print(correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
    if i % 100 == 0 :

        print(sess.run(accuracy, feed_dict ={x:mnist.test.images, y_:mnist.test.labels}))