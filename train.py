import os
import numpy as np
import tensorflow as tf
import pandas as pd
from DataProcess import get_data
from model import cnn_lstm
from keras.models import load_model
from sklearn.metrics import roc_auc_score
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam


def calculate_performace(test_num, pred_y,  labels):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if labels[index] == 1:
            if labels[index] == pred_y[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if labels[index] == pred_y[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    acc = float(tp + tn) / test_num
    precision = float(tp) / (tp + fp)
    sensitivity = float(tp) / (tp + fn)
    specificity = float(tn) / (tn + fp)
    MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return acc, precision, sensitivity, specificity, MCC


def generator_data(pos, neg, b_size):
    while True:
        index_pos = np.random.choice(pos.shape[0], int(b_size * 0.2), replace=False)
        index_neg = np.random.choice(neg.shape[0], int(b_size * 0.8), replace=False)
        x = np.vstack((pos[index_pos], neg[index_neg]))
        y = [1 for _ in range(len(index_pos))]+[0 for _ in range(len(index_neg))]
        index = np.random.choice(x.shape[0], x.shape[0], replace=False)
        x = x[index]
        y = np.array(y)[index]

        yield x, y


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.Session(config=config)
    tf.keras.backend.set_session(sess)

    input_dim, input_length, n_filter = 4, 21, 128
    window_size = input_length
    # rbp_name = "FXR1"
    batch_size, epochs = 128, 500
    f_auc = open('./save_auc.txt', 'w')
    name, Aucs = [], []
    # 'C17ORF85', 'QKI', 'WTAP', 'EWSR1', 'AGO1', 'AGO2', 'AGO3', 'AUF1'
    # for rbp_name in ['QKI']:
    for rbp_name in sorted(os.listdir('/data/wuhehe/wuhe/Bsite_data')):
        print("rbp name: ", rbp_name)
        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True)
        model = cnn_lstm(input_dim, input_length, n_filter)
        model.compile(loss='binary_crossentropy', optimizer=sgd)  # 'rmsprop')Adam(lr=1e-4)
        # model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-2))
        data_pos, data_neg = get_data(window_size=window_size, rbp_name=rbp_name, train=True)
        train_pos, train_neg = data_pos[:int(0.8*len(data_pos))], data_neg[:int(0.8*len(data_neg))]
        val_pos, val_neg = data_pos[int(0.8*len(data_pos)):], data_neg[int(0.8*len(data_neg)):]
        print('len(data_pos): {} len(data_neg): {}'.format(len(data_pos), len(data_neg)))
        earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=0)
        # steps_per_epoch = len(Y_train) // batch_size, callbacks=[earlystopper],
        history = model.fit_generator(generator_data(pos=train_pos, neg=train_neg, b_size=batch_size),
                                      verbose=2, epochs=epochs, callbacks=[earlystopper],
                                      validation_data=generator_data(pos=val_pos, neg=val_neg, b_size=batch_size),
                                      steps_per_epoch=int(0.005 * len(train_pos)), validation_steps=30)

        model.save('./models/{}.h5'.format(rbp_name))
        # model = load_model('./models/{}.h5'.format(rbp_name))
        loss_train = history.history['loss']
        loss_val = history.history['val_loss']
        f_loss = open('./{}_loss.txt'.format(rbp_name), 'w')
        for i in range(len(loss_train)):
            f_loss.writelines([str(i), "   "+"loss_train: "+str(loss_train[i])+"    loss_val: "+str(loss_val[i])+'\n'])
        f_loss.close()

        test_pos, test_neg = get_data(window_size=window_size, rbp_name=rbp_name, train=False)
        test_data = np.vstack((test_pos, test_neg))
        test_label = np.array([1. for _ in range(len(test_pos))] + [0. for _ in range(len(test_neg))])

        preds = model.predict(test_data)
        auc = roc_auc_score(y_true=test_label, y_score=preds)
        name.append(rbp_name)
        Aucs.append(auc)
        f_auc.writelines([str(rbp_name), ": ", str(auc), '\n'])
        print("rbp name: {}, AUC: {}".format(rbp_name, auc))
    # f_loss.close()
    _save = pd.DataFrame({'name': name, 'auc': Aucs})
    _save.to_csv('all_auc.csv', index=False)
    f_auc.close()
