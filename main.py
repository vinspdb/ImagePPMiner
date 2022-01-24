import numpy as np
from keras.utils import np_utils
from sklearn import preprocessing
from keras.layers import Conv2D, Activation
from keras import regularizers
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import MaxPooling2D
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping
from keras.layers.normalization import BatchNormalization
from ImagePPMiner import ImagePPMiner
import sys
seed = 123
np.random.seed(seed)
from tensorflow import set_random_seed
set_random_seed(seed)
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNN for next activity prediction.')

    parser.add_argument('-event_log', type=str, help="Event log name")
    parser.add_argument('-n_layers', type=int, help="Number of convolutional layers")

    args = parser.parse_args()

    dataset = args.event_log
    n_layer = args.n_layers
    pm = ImagePPMiner(dataset)
    log = pm.import_log()
    max_trace, n_caseid, n_activity = pm.dataset_summary(log=log)
    train_act, train_temp, test_act, test_temp = pm.generate_prefix_trace(log=log, n_caseid=n_caseid)
    X_train = pm.generate_image(act_val=train_act, time_val=train_temp, max_trace=max_trace, n_activity=n_activity)
    X_test = pm.generate_image(act_val=test_act, time_val=test_temp, max_trace=max_trace, n_activity=n_activity)

    l_train = pm.get_label(train_act)
    l_test = pm.get_label(test_act)

    le = preprocessing.LabelEncoder()
    l_train = le.fit_transform(l_train)
    l_test = le.transform(l_test)
    num_classes = le.classes_.size

    X_train = np.asarray(X_train)
    l_train = np.asarray(l_train)

    X_test = np.asarray(X_test)
    l_test = np.asarray(l_test)

    train_Y_one_hot = np_utils.to_categorical(l_train, num_classes)
    test_Y_one_hot = np_utils.to_categorical(l_test, num_classes)

    ##############neural network architecture##############
    model = Sequential()
    reg = 0.0001
    input_shape = (max_trace, n_activity, 2)

    if int(n_layer) == 1:
        model.add(Conv2D(32, (2, 2), input_shape=input_shape, padding='same', kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    elif int(n_layer) == 2:
        model.add(Conv2D(32, (2, 2), input_shape=input_shape, padding='same', kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, (4, 4), padding='same', kernel_regularizer=regularizers.l2(reg), ))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    elif int(n_layer) == 3:
        model.add(Conv2D(32, (2, 2), input_shape=input_shape, padding='same', kernel_initializer='glorot_uniform',
                         kernel_regularizer=regularizers.l2(reg)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(64, (4, 4), padding='same', kernel_regularizer=regularizers.l2(reg), kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

        model.add(Conv2D(128, (8, 8), padding='same', kernel_regularizer=regularizers.l2(reg), kernel_initializer='glorot_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))


    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', name='act_output'))

    print(model.summary())

    opt = Nadam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004, clipvalue=3)
    model.compile(loss={'act_output': 'categorical_crossentropy'}, optimizer=opt, metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    history = model.fit(X_train, {'act_output': train_Y_one_hot}, validation_split=0.2, verbose=2, callbacks=[early_stopping], batch_size=128, epochs=500)

    y_pred_test = model.predict(X_test)
    max_y_pred_test = np.argmax(y_pred_test, axis=1)
    max_y_test = np.argmax(test_Y_one_hot, axis=1)
    print(classification_report(max_y_test, max_y_pred_test, digits=3))
