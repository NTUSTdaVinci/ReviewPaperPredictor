from PyQt5.QtCore import QThread, pyqtSignal

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
# ====== For Pyinstaller =====
import sklearn.utils._cython_blas
import sklearn.utils._typedefs
import sklearn.neighbors._partition_nodes
import sklearn.tree
import sklearn.tree._utils
import scipy
# ============================
from keras.models import Sequential
from keras import callbacks, layers

from keyword_calculator import KeywordCalculator

import matplotlib.pyplot as plt

NUM_EPOCHS = 5

class CustomCallback(callbacks.Callback):
    def __init__(self):
        self.task_type = ''
        self.epoch = 1
        self.batch = 0
        self.epoch_time_start = 0
        self.epoch_max = NUM_EPOCHS

    def on_epoch_begin(self, epoch, logs=None):
        #print("on_epoch_begin")
        self.epoch = epoch
        self.epoch_time_start = time.time()
        epoch_counter = epoch + 1
        #msg_IO('== Epoch {counter}/{ep_max} ==\n'.format(counter=epoch_counter, ep_max=self.epoch_max))

    def on_epoch_end(self, epoch, logs=None):
        # msg_IO(' - time: {:7.0f}s\n'.format(time.time() - self.epoch_time_start))
        # msg_IO(' - loss: {:7.4f}'.format(logs["loss"]))
        # msg_IO(' - accuracy: {:7.4f}\n'.format(logs["accuracy"]))
        # msg_IO(' - val_loss: {:7.4f}'.format(logs["val_loss"]))
        # msg_IO(' - val_accuracy: {:7.4f}\n'.format(logs["val_accuracy"]))
        # msg_IO('\n')
        #print("on_epoch_end")
        a = 1

class BuildingTask(QThread):
    qthread_signal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, ip, wp, op, tc, ac, lc):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.input_path = ip
        self.wb_path = wp
        self.output_path = op
        self.ti_col = tc
        self.ab_col = ac
        self.label_col = lc

    def run(self):
        # read training data from input file: title, abstract, label
        train = pd.read_excel(self.input_path, usecols=[self.ti_col, self.ab_col, self.label_col])
        print(self.ti_col+", "+self.ab_col+", "+self.label_col+"\n")
        print(train[self.ti_col])

        train[self.ti_col] = train[self.ti_col].astype(str)
        train[self.ab_col] = train[self.ab_col].astype(str)

        print("Obtain " + str(len(train[self.ti_col])) + " data\n")
        self.qthread_signal.emit("Obtain " + str(len(train[self.ti_col])) + " data\n")

        # read word bank
        df_wb = pd.read_excel(self.wb_path)

        print("'wbList'")
        keyword_counter = KeywordCalculator(train[self.ti_col].values, train[self.ab_col].values, df_wb)
        input_data = keyword_counter.get_result()
        self.qthread_signal.emit("Finish keyword count\n")

        # spilt keywords statistics into training(75%) and testing(25%) data
        merger_train, merger_val, \
        y_train, y_val \
            = train_test_split(input_data, train[self.label_col].values, test_size=0.25, random_state=1000)

        self.qthread_signal.emit("merge Train on " + str(len(merger_train)) + " samples,")
        self.qthread_signal.emit("merge validate on " + str(len(merger_val)) + " samples\n")

        # print(input_ti_train)
        self.qthread_signal.emit("Constructing AI model...\n")

        # create and setup model
        model = Sequential()

        # setup layers
        if len(df_wb['keyword']) > 50:
            model.add(layers.Dense(256, activation='relu'))
            model.add(layers.Dense(128, activation='relu'))
        elif len(df_wb['keyword']) > 30:
            model.add(layers.Dense(128, activation='relu'))

        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        self.qthread_signal.emit("Training AI model...\n")
        history = model.fit(x=merger_train,
                            y=y_train,
                            epochs=5,
                            verbose=2,
                            validation_data=(merger_val, y_val),
                            batch_size=50,
                            callbacks=[CustomCallback()])

        self.qthread_signal.emit("Finish AI model training\n")
        model.summary()

        self.qthread_signal.emit("Saving output files...\n")
        # save the keywords statistics result in output folder
        keyword_counter.save_statistics(self.output_path)

        # save model in output folder
        model.save(self.output_path + '/model.h5')

        self.qthread_signal.emit("Saving history files...\n")
        self.print_history(history)
        self.qthread_signal.emit("Finished")
        self.finished.emit()

    def print_history(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'ro', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(self.output_path + '/acc.png')

        plt.figure()

        plt.plot(epochs, loss, 'ro', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()

        # plt.show()
        print("save dir: " + self.output_path)
        plt.savefig(self.output_path + '/loss.png')

