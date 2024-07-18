import numpy as np
import tflearn
from keras.layers import LSTM, Dense
from tensorflow.python.framework import ops
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data
from tflearn.layers.estimator import regression

from Evaluation import evaluation


def Model(X, Y, test_x, test_y, sol):
    LR = 1e-3
    IMG_SIZE = 20
    ops.reset_default_graph()
    activation = ['Relu', 'linear', 'tanh', 'sigmoid']
    optimizer = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad']
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, name='layer-conv1', activation=activation[int(sol[0])])  # activation='linear'
    convnet = max_pool_2d(convnet, 5)

    convnet_2 = conv_2d(convnet, 64, 5, name='layer-conv2', activation=activation[int(sol[0])])
    convnet = max_pool_2d(convnet_2, 5)

    convnet = conv_2d(convnet, 128, 5, name='layer-conv3', activation=activation[int(sol[0])])
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, name='layer-conv4', activation=activation[int(sol[0])])
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, name='layer-conv5', activation=activation[int(sol[0])])
    convnet = max_pool_2d(convnet, 5)

    convnet1 = LSTM(convnet, 50,  input_shape=(1, 200))
    convnet2 = Dense(convnet1, Y.shape[1])
    regress = regression(convnet2, optimizer=optimizer[int(sol[1])], learning_rate=int(sol[3]),   #   learning_rate=0.01
                         loss='mean_square', name='target')   # optimizer = 'sgd'

    model = tflearn.DNN(regress, tensorboard_dir='log')

    MODEL_NAME = 'test.model'.format(LR, '6conv-basic')
    model.fit({'input': X}, {'target': Y}, n_epoch=int(sol[2]),   # n_epoch= 5
              validation_set=({'input': test_x}, {'target': test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    pred = model.predict(test_x)
    return pred


def Model_OMCN_LSTM(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [0, 0, 1, 0.01]
    IMG_SIZE = 20
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    pred_1 = Model(Train_X, train_target, Test_X, test_target,sol)
    pred_2 = Model(Train_X, train_target, Test_X, test_target,sol)
    pred_3 = Model(Train_X, train_target, Test_X, test_target,sol)

    pred = (pred_1 + pred_2 + pred_3) / 3
    Eval = evaluation(pred, test_target)
    return Eval, pred

# if __name__ == '__main__':
#     a = np.random.random((100, 10))
#     b = np.random.randint(low=0, high=2, size=(100, 1))
#     c = np.random.random((100, 10))
#     d = np.random.randint(low=0, high=2, size=(100, 1))
#     Eval, pred = Model_OMCN_LSTM(a, b, c,d)
#     sfds = 6