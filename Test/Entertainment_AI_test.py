#-*- coding: utf-8 -*-
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


input_data = [[1,5,3,7,8,10,12],
              [2,3,4,5,6,11,19]]  # 핵심어에 대한 입력을 메트릭스로 변환 했다 가정
label_data = [[0,0,0,1,0],
              [1,0,0,0,0]]        # 결과를 5개의 감정 중 하나만 찾아 낸다 가정

# NN 멀티로 구성 한다 가정 하고 몇가지 상수 정의
# 왜 상수 인데 tf.constants 로 정의 안하냐고 생각 할수 있게지만, 그래프로 만들 상수가 아니여서 일반 상수로 했다 생각 하면됨
INPUT_SIZE = 7
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES = 5
Learning_rate = 0.1

# 일반적인 가설 모델인 Wx +  B  라 가정 하고
x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, CLASSES])

# placeholder  는 맵핑을 해주어야 함
# place 는 주로 일반적인 입력 데이터를 tensor 로 변환 하기 위해서 사용하고 변환 과정 전에 아래 처럼 맴핑을 한번 해줌
tensor_map = {x : input_data, y_ : label_data}

# 입력 레이어에 대한 가중치 모델링
# truncated_normal 은 가중치 값을 일반적인 확률분포를 이용해서 주겠다는 의미
# W 와 b 는 Variable 이다
W_h1 = tf.Variable(tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]))
b_h1 = tf.Variable(tf.zeros(shape=[HIDDEN1_SIZE]), dtype=tf.float32) # 디멘전이 2개인데 하나로 해보기

W_h2 = tf.Variable(tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]))
b_h2 = tf.Variable(tf.zeros(shape=[HIDDEN2_SIZE]), dtype=tf.float32) # 디멘전이 2개인데 하나로 해보기

W_o = tf.Variable(tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]))
b_o = tf.Variable(tf.zeros(shape=[CLASSES]), dtype=tf.float32)

# 로지스틱 이기 때문에 시그모이드 사용 한다
hidden1 = tf.sigmoid(tf.matmul(x, W_h1) + b_h1)   # softmax 는 sigmod 를 정규화한 모델
hidden2 = tf.sigmoid(tf.matmul(hidden1, W_h2) + b_h2)
y = tf.sigmoid(tf.matmul(hidden2, W_o) + b_o)

# 코스트 모듈
cost = tf.reduce_mean(-y_*tf.log(y)-(1-y)*tf.log(1-y)) ## 전체 cost 를 평균낸값
#cost = -tf.reduce_sum(y_*tf.log(y))

# 최소화 적용
train = tf.train.GradientDescentOptimizer(Learning_rate)

init = tf.initialize_all_variables()
## 여기 까지가 모델 설계 끝난것임
## 다음은 훈련을 하면됨
sess = tf.Session()
sess.run(init)
print sess.run(cost, feed_dict=tensor_map)