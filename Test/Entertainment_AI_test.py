# -*- coding: utf-8 -*-

import tensorflow as tf


# 리니어 리그리션 에서는 민 스퀘어
# 로지스틱 에서는 크로스 엔트로피 사용

# : 일반 입력 데이터 가지고 오기
# : 지도 학습 이라면 레이블 데이터 까지 함께 가지고 오기
# : 입력 데이터를 placeholder 처리 & 맵핑
# : 뉴럴 네트워크를 단일로 구성 할지 멀티로 구성 할지 결정 하고 멀티 라면 몇개의 히든 레이어를 둘지 결정
# : 입력 레이어에 대한 가중치를 어떤 확률분포를 사용 할지 결정 하고 적용
# : 입력 레이어에 해단 가중치가 모델릴 했다면 그래프 적용


# 입력 데이터에 해당 하는 부분, placeholder 처리 해야할 대상
# 1 2 3 4 5 6 7 8 9 10
# 1 2 3 4 5 6 7 8 9 10
# ......
# 10*10 으로 가정한 벡터 에서 4등분한 4개의 영역을 서로 다른 감정 으로 할당하고
# 입력받은 N 개의 핵심어를 각 4개의 영역에 적절한 감정 지수로 데이터를 할당
# 해당 입력의 감정 결과를 레이블로 제공
# 훈련 => test

"""
단어 100 개로 한정, 결과는 댄스, 트로트, 힙합, 발라드 4개로 한정

* 힙합
비와이, 씨잼, G2, 쌈디, 서출구, 키썸, 딘딘, 일레븐, 오케이션, 일리닛, 어글리덕, 그레이, 팔로알토, 비프리, 이센스, 더콰이엇, 오왼오바도즈, 레디, 로꼬, 기리보이, 크러쉬, 에픽하이, 드렁큰타이거, 다듀

* 발라드
이루, 테이, KCM, 이승기, 이기찬, 에반, 박완규, 휘성, 플라이투더스카이, 버즈, SG워너비, 백지영, 린, 장리인, 양파, 이소은, 성시경, 신승훈, 임재범, 변진섭, 이소라, 제이, 서지원, 김돈규, 박상민

* 댄스
보아, 채연, 이효리, 손담비, 지나, NS윤지, 아이유, 에일리, 걸스데이, 시크릿, 티아라, 디유닛, 씨스타, 미쓰에이, 투애니원, 브라운아이드걸스, 글램, 쇼콜라, 타이니지, 박재범, 싸이, 서인국, 세븐, 김현중, 샤이니

* 트로트
장윤정, 홍진영, 진성, 진미령, 오승근, 신유, 조항조, 금잔디, 유진표, 남진, 진시몬, 김성환, 김용임, 박상철, 추가열, 박구윤, 진성, 이용, 최진희, 심수봉, 나훈아, 이미자, 이애란, 태진아, 유지나
단어 100 개로 한정, 결과는 댄스, 트로트, 힙합, 발라드 4개로 한정



"""

# [힙합, 발라드, 댄스 트로트] 피처에 대한 입력을 메트릭스로 변환피처에 대한 입력을 메트릭스로 변환

input_data = [
    [1, 1, 1, 1, 0, # 힙합
     0, 0, 0, 0, 0, # 발라드
     0, 0, 0, 0, 0, # 댄스
     0, 0, 0, 0, 0], # 트로트



    # -----------------------------------------------------------------------------

    [0, 0, 0, 0, 0,
     1, 1, 1, 1, 0,
     0, 0, 0, 0, 0,
     0, 0, 0, 0, 0]



    # -----------------------------------------------------------------------------


    # -----------------------------------------------------------------------------

]


label_data = [  [1, 0, 0, 0],


                [0, 1, 0, 0]




              ]

# NN 멀티로 구성 한다 가정 하고 몇가지 상수 정의
# 왜 상수 인데 tf.constants 로 정의 안하냐고 생각 할수 있게지만, 그래프로 만들 상수가 아니여서 일반 상수로 했다 생각 하면됨
INPUT_SIZE = 20
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES = 4
Learning_rate = 0.2

# 일반적인 가설 모델인 Wx +  B  라 가정 하고
x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE], name='x')
y_ = tf.placeholder(tf.float32, shape=[None, CLASSES], name='y_')

# placeholder  는 맵핑을 해주어야 함
# place 는 주로 일반적인 입력 데이터를 tensor 로 변환 하기 위해서 사용하고 변환 과정 전에 아래 처럼 맴핑을 한번 해줌
tensor_map = {x: input_data, y_: label_data}

# 입력 레이어에 대한 가중치 모델링
# truncated_normal 은 가중치 값을 일반적인 확률분포를 이용해서 주겠다는 의미
# W 와 b 는 Variable 이다
W_h1 = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32, name='W_h1')
b_h1 = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32, name='b_h1')  # 디멘전이 2개인데 하나로 해보기

W_h2 = tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32, name='W_h2')
b_h2 = tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]), dtype=tf.float32, name='b_h2')  # 디멘전이 2개인데 하나로 해보기

## 실제로는 중간에 dropout, 필터, 패딩 등을 적용 , 여기서는 간단한 모델링 이므로 적용 하지 않음

W_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]), dtype=tf.float32, name='W_o')
b_o = tf.Variable(tf.zeros(shape=[CLASSES]), dtype=tf.float32, name='b_o')

#saver 레이어
param_list = [W_h1, b_h1, W_h2, b_h2, W_o, b_o]  #
saver = tf.train.Saver(param_list)

# 로지스틱 이기 때문에 시그모이드 사용 한다
hidden1 = tf.sigmoid(tf.matmul(x, W_h1) + b_h1, name='hidden1')  # softmax 는 sigmod 를 정규화한 모델
hidden2 = tf.sigmoid(tf.matmul(hidden1, W_h2) + b_h2, name='hidden2')

y = tf.sigmoid(tf.matmul(hidden2, W_o) + b_o, name='y')  # 왜 softmax 를 사용 하지 않고 sigmoid 를 적용 했냐 하면, 근간이 되는 요소를 노출 하기 위해서 이다
                                                # softmax 내부가 sigmoid 를 더 가공한것에 불과 하다

# 코스트 모듈 (선택 해서 사용 하면됨)
#cost = tf.reduce_mean(-y_ * tf.log(y) - (1 - y) * tf.log(1 - y),name='cost')  ## 전체 cost 를 평균낸값
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# cost = -tf.reduce_sum(y_*tf.log(y))

# 최소화 적용
train = tf.train.GradientDescentOptimizer(Learning_rate).minimize(cost)

init = tf.initialize_all_variables()
## 여기 까지가 모델 설계 끝난것임
## 다음은 훈련

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess = tf.Session()


sess.run(init)
for i in range(1000) :
    _, cost_rate, acc, yy_, o_ = sess.run([train,cost, accuracy, y, W_o],  feed_dict=tensor_map)
    pred = sess.run(tf.argmax(y, 1), tensor_map)
    if i % 100 == 0:
        print "Train retry  : ", i
        print "Cost : ", cost_rate
        print "Accuracy : ", acc * 100 , " %"
        saver.save(sess, '/Users/Naver/tensorflow_save/save')
        print "-----------------------------------"

sess.close()



# loss 가 줄어 드는건 결국   W 와 b 를 최적화된 값을 찾아 가고 있다는 의미 인데
# 최적화된 값을 찾아 가는 핵심 로직은 backpropagation 이 해주는 것이다
# 이부분을 tensorflow 가 처리 해주기 때문에 어렵지 않게 가능



