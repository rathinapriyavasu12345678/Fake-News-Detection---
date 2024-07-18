import numpy as np
import tflearn
from tensorflow.python.framework import ops
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

from Evaluation import evaluation


def Model(X, Y, test_x, test_y):
    LR = 1e-3
    IMG_SIZE = 20
    ops.reset_default_graph()
    activation = ['Relu', 'linear', 'tanh', 'sigmoid']
    optimizer = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad']
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, name='layer-conv1', activation='linear') # activation= activation[int(sol[0])]
    convnet = max_pool_2d(convnet, 5)

    convnet_2 = conv_2d(convnet, 64, 5, name='layer-conv2', activation='linear')
    convnet = max_pool_2d(convnet_2, 5)

    convnet = conv_2d(convnet, 128, 5, name='layer-conv3', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, name='layer-conv4', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, name='layer-conv5', activation='linear')
    convnet = max_pool_2d(convnet, 5)

    convnet1 = fully_connected(convnet, 1024, name='layer-conv', activation='linear')
    convnet2 = dropout(convnet1, 0.8)

    convnet3 = fully_connected(convnet2, Y.shape[1], name='layer-conv-before-softmax', activation='linear')

    regress = regression(convnet3, optimizer='sgd', learning_rate=0.01,
                         loss='mean_square', name='target') # optimizer = optimizer[int(sol[1])]

    model = tflearn.DNN(regress, tensorboard_dir='log')

    MODEL_NAME = 'test.model'.format(LR, '6conv-basic')
    model.fit({'input': X}, {'target': Y}, n_epoch=5,  # n_epoch= int[sol[o]]
              validation_set=({'input': test_x}, {'target': test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    pred = model.predict(test_x)

    return pred


def Model_CNN(train_data, train_target, test_data, test_target):
    IMG_SIZE = 20
    Train_X = np.zeros((train_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(train_data.shape[0]):
        temp = np.resize(train_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Train_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))

    Test_X = np.zeros((test_data.shape[0], IMG_SIZE, IMG_SIZE, 1))
    for i in range(test_data.shape[0]):
        temp = np.resize(test_data[i], (IMG_SIZE * IMG_SIZE, 1))
        Test_X[i] = np.reshape(temp, (IMG_SIZE, IMG_SIZE, 1))
    pred = Model(Train_X, train_target, Test_X, test_target)

    Eval = evaluation(pred, test_target)
    return Eval, pred

# if __name__ == '__main__':
#     a = np.random.random((100, 10))
#     b = np.random.randint(low=0, high=2, size=(100, 1))
#     c = np.random.random((100, 10))
#     d = np.random.randint(low=0, high=2, size=(100, 1))
#     Eval, pred = Model_CNN(a, b, c,d)
# #     sfds = 6