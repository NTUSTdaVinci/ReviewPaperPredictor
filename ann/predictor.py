from PyQt5.QtCore import QThread, pyqtSignal

import time
import numpy as np
import pandas as pd

from keras.models import load_model, Sequential

from ann.keyword_calculator import KeywordCalculator

import matplotlib.pyplot as plt


class PredictionTask(QThread):
    qthread_signal = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, ip, wp, mp, op, tc, ac):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.input_path = ip
        self.wb_path = wp
        self.model_path = mp
        self.output_path = op
        self.ti_col = tc
        self.ab_col = ac

    def run(self):
        # read training data from input file: title, abstract, label
        train = pd.read_excel(self.input_path)

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

        self.qthread_signal.emit("Loading model...\n")
        model = load_model(self.model_path)

        self.qthread_signal.emit("Predicting...\n")
        predict = model.predict(input_data)

        self.qthread_signal.emit("Saving output files...\n")
        # save the keywords statistics result in output folder
        keyword_counter.save_statistics(self.output_path)

        col_name = train.columns.tolist()
        col_name.insert(0, 'Possibility of Review Paper')
        training_input = train.reindex(columns=col_name)
        training_input['Possibility of Review Paper'] = predict
        input_file_name = self.input_path.split('/')[-1].split('.')[0]
        output_file_name = input_file_name + '_' + str(int(time.time()))
        training_input.to_excel(self.output_path + '/' + output_file_name + '.xlsx', index=False)

        self.qthread_signal.emit("Finished")
        self.finished.emit()