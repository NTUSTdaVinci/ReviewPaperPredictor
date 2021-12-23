import tkinter as tk
from tkinter import filedialog
import threading

import numpy as np
import pandas as pd

from keras import callbacks
from keras import Input
from keras import layers
from keras import regularizers
from keras.models import Sequential, Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

import time
import os
import re

import concurrent.futures
import logging
import matplotlib.pyplot as plt

from threading import Thread

MAX_LENGTH = 450
NUM_EMBEDDING_DIM = 100
NUM_LSTM_UNITS = 96
TEXT_LINE_NO = 1

HISTORY_FLAG = False

class Threader(threading.Thread):
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.daemon = True
        self.start()

    def run(self):
        file_check()


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args,
                                        **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


class CustomCallback(callbacks.Callback):
    def __init__(self):
        self.task_type = ''
        self.epoch = 1
        self.batch = 0
        self.epoch_time_start = 0

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.epoch_time_start = time.time()
        epoch_counter = epoch + 1
        msg_IO('== Epoch {}/10 ==\n'.format(epoch_counter))

    def on_epoch_end(self, epoch, logs=None):
        msg_IO(' - time: {:7.0f}s\n'.format(time.time() - self.epoch_time_start))
        msg_IO(' - loss: {:7.4f}'.format(logs["loss"]))
        msg_IO(' - accuracy: {:7.4f}\n'.format(logs["accuracy"]))
        msg_IO(' - val_loss: {:7.4f}'.format(logs["val_loss"]))
        msg_IO(' - val_accuracy: {:7.4f}\n'.format(logs["val_accuracy"]))
        msg_IO('\n')


def file_check():
    print("run AI")
    file_path = path_entry.get()
    ti_name = col_entry.get()
    ab_name = ab_entry.get()
    label_name = label_entry.get()
    save_dir = folder_entry.get()
    wb_path = wb_entry.get()

    # rewrite default file
    with open('default_value.txt', 'w+') as f:
        f.write(file_path+"\n")
        f.write(wb_path+"\n")
        f.write(save_dir+"\n")
        f.write(ti_name+"\n")
        f.write(ab_name+"\n")
        f.write(label_name)

    # check files or directory exist
    if os.path.isfile(file_path) and os.path.isfile(wb_path) and os.path.isdir(save_dir):
        build_ai(file_path, ti_name, ab_name, label_name, save_dir, wb_path)
    else:
        msg_IO("Files or directory do not exist.")

def print_train_test_data(x_train, y_train, x_test, y_test):
    print("\nTraining Set")
    print("-" * 10)
    print(f"x_train: {x_train.shape}")
    print(f"y_train : {y_train.shape}")

    print("-" * 10)
    print(f"x_val:   {x_test.shape}")
    print(f"y_val :   {y_test.shape}")
    print("-" * 10)
    print("Test Set\n")


def print_history(history, save_dir):
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
    plt.savefig(save_dir + '/acc.png')

    plt.figure()

    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    # plt.show()
    print("save dir" + save_dir)
    plt.savefig(save_dir + '/loss.png')

# Review paper ai builder
def build_ai(filePath, col_name, ab_name, label_name, save_dir, wb_path):
    msg_IO('Reading data ...\n')

    # read word bank
    df_wb = pd.read_excel(wb_path)
    wbList = df_wb.values.tolist()
    print(wbList)

    # read training data: title, abstract, label
    train = pd.read_excel(filePath, usecols=[col_name, ab_name, label_name])
    train[col_name] = train[col_name].astype(str)
    train[ab_name] = train[ab_name].astype(str)

    input = []
    input_counter = 0

    msg_IO("Obtain "+ str(len(train[col_name])) +" data\n")
    msg_IO("Calculating keyword count in training data.\n")
    print("Calculating keyword count in training data ...")
    for ti_sentence, ab_sentence in zip(train[col_name].values, train[ab_name].values):
        ti_list = []
        ab_list = []
        for word in wbList:
            ti_list.append(len(re.findall(word[0].lower(), str(ti_sentence).lower(), re.IGNORECASE)))
            ab_list.append(len(re.findall(word[0].lower(), str(ab_sentence).lower(), re.IGNORECASE)))

        # print(ti_list)
        # print(ab_list)
        sentence_list = ti_list + ab_list
        #print(len(sentence_list))
        ti_list.clear()
        ab_list.clear()
        #input = np.vstack((input, sentence_list))
        #input = [[input], [sentence_list]]
        input.append(sentence_list)
        input_counter+=1

        # if input_counter == 0:
        #     msg_IO('1/'+str(len(train[col_name]))+' finished.\n')
        # else:
        #     msg_IO_line(str(input_counter)+'/'+str(len(train[col_name]))+' finished.\n')
        #     time.sleep(0.5)

    print("****" + str(len(input)))
    print(input[5])
    input_data = np.array(input)
    col = np.append(wbList, wbList)
    print(len(col))
    df_word = pd.DataFrame(input, columns=col)
    df_word.to_excel("keywords statistics.xlsx")

    msg_IO('Read ' + str(len(train[col_name])) + ' data \n')

    # Train-test split
    merger_train, merger_val, \
    y_train, y_val \
        = train_test_split(input_data, train[label_name].values, test_size=0.25, random_state=1000)

    msg_IO("merge Train on " + str(len(merger_train)) + " samples,")
    msg_IO("merge validate on " + str(len(merger_val)) + " samples\n")

    # print(input_ti_train)
    msg_IO("Constructing AI model...\n")
    print(input[5])
    print(merger_train[5])

    model = Sequential()

    if len(wbList) > 30:
        model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    msg_IO("Training AI model...\n")
    history = model.fit(x=merger_train,
                        y=y_train,
                        epochs=10,
                        verbose=2,
                        validation_data=(merger_val, y_val),
                        batch_size=50,
                        callbacks=[CustomCallback()])
    model.summary()

    model.save(save_dir + '/model.h5')

    print_history(history, save_dir)


def browseFolder():
    tk.Tk().withdraw()

    folder_entry.delete(0, 'end')
    folder_entry.insert(0, filedialog.askdirectory())

def browseDataFile():
    tk.Tk().withdraw()

    path_entry.delete(0, 'end')
    path_entry.insert(0, filedialog.askopenfilename())

def browseWBFile():
    tk.Tk().withdraw()

    wb_entry.delete(0, 'end')
    wb_entry.insert(0, filedialog.askopenfilename())

def msg_IO(msg):
    msg_text.insert('end', msg)
    print(msg)
    msg_text.update_idletasks()
    global TEXT_LINE_NO
    TEXT_LINE_NO+=1

def msg_IO_line(msg):
    #msg_text.delete(2.0, 'end')
    msg_text.delete('current linestart', 'current lineend+1c')
    msg_text.insert('end', msg)
    msg_text.update_idletasks()
    #print(float(TEXT_LINE_NO-1))


if __name__ == '__main__':

    FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.DEBUG, filename='myLog.log', filemode='w', format=FORMAT)

    logging.debug('debug message')
    logging.info('info message')
    logging.warning('warning message')
    logging.error('error message')
    logging.critical('critical message')

    # Flag for default file:
    # True: exist, False: not exist
    DEFAULT_FILE = True
    # Check default file exist
    if os.path.isfile('default_value.txt'):
        with open('default_value.txt', 'r') as f:
            default_str = f.read().splitlines()
            print(default_str)

        # Check file format
        if len(default_str) != 6:
            DEFAULT_FILE = False
    else:
        DEFAULT_FILE = False

    # Draw UI
    window = tk.Tk()
    window.title('AI Builder for Review Paper')
    window.geometry('800x800')
    window.configure(background='white')

    header_label = tk.Label(window,
                            text='\nBuild your AI for Review Paper',
                            bg='white',
                            font=('Times New Roman', 26))
    header_label.grid(row=0, column=0)

    top_frame = tk.Frame(window)
    top_frame.grid(row=1, column=0)
    content_label = tk.Label(top_frame,
                             text='\nThis program generate the review paper AI of input data',
                             bg='white',
                             font=('Times New Roman', 14))
    content_label.pack(side=tk.TOP)

    center_frame = tk.Frame(window, bg='white')
    center_frame.grid(row=2, column=0)
    Step_headline_label = tk.Label(center_frame,
                                   text='\n\nSteps.',
                                   bg='white',
                                   font=('Times New Roman', 14))
    Step_headline_label.pack(side=tk.TOP)
    Step_num_label = tk.Label(center_frame,
                              text='',
                              bg='white',
                              font=('Times New Roman', 12),
                              anchor='ne',
                              height='5')
    Step_num_label.pack(side=tk.LEFT)
    Step_label = tk.Label(center_frame,
                          text='Choose your xlsx file by pressing "Browse file" button.\n'
                               'Enter column name of data stream and label in file. \nPress Execute button.\n'
                               '\nPLEASE CLOSE any output file the program generates before executing',
                          bg='white',
                          font=('Times New Roman', 12),
                          justify='left',
                          anchor='nw',
                          wraplength='350')
    Step_label.pack(side=tk.RIGHT)

    line_frame = tk.Frame(window, bg='white', bd='0px')
    line_frame.grid(row=3, column=0)
    cv = tk.Canvas(line_frame, bg='white', bd=0, height='70', width='500', highlightthickness=0)
    cv.pack(side=tk.TOP)
    line = cv.create_line(0, 25, 500, 25)

    bottom_frame = tk.Frame(window, bg='white', bd='0px')
    bottom_frame.grid(row=4, column=0)
    Step_label = tk.Label(bottom_frame,
                          text='Input file path',
                          bg='white',
                          font=('Times New Roman', 10),
                          width='15',
                          anchor='nw')
    Step_label.grid(row=0, column=0, sticky='E')
    path_entry = tk.Entry(bottom_frame, width='40', font=('Times New Roman', 12))

    # if default file exist, insert default value
    if DEFAULT_FILE:
        path_entry.insert(0, default_str[0])
    path_entry.grid(row=0, column=1)

    parse_files_btn = tk.Button(bottom_frame, text='Browse file', anchor='nw', command=browseDataFile)
    parse_files_btn.grid(row=0, column=2)

    wb_label = tk.Label(bottom_frame,
                          text='Word bank path',
                          bg='white',
                          font=('Times New Roman', 10),
                          width='15',
                          anchor='nw')
    wb_label.grid(row=1, column=0, sticky='E')
    wb_entry = tk.Entry(bottom_frame, width='40', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        wb_entry.insert(0, default_str[1])
    wb_entry.grid(row=1, column=1)

    wb_files_btn = tk.Button(bottom_frame, text='Browse file', anchor='nw', command=browseWBFile)
    wb_files_btn.grid(row=1, column=2)

    Output_frame = tk.Frame(window, bg='white', bd='0px')
    Output_frame.grid(row=5, column=0)
    folder_label = tk.Label(Output_frame,
                            text='Output folder path',
                            bg='white',
                            font=('Times New Roman', 10),
                            width='63',
                            anchor='nw')
    folder_label.pack(side=tk.TOP)
    folder_entry = tk.Entry(Output_frame, width='50', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        folder_entry.insert(0, default_str[2])
    folder_entry.pack(side=tk.LEFT)

    parse_folder_btn = tk.Button(Output_frame, text='Browse folder', anchor='nw', command=browseFolder)
    parse_folder_btn.pack(side=tk.RIGHT)

    col_frame = tk.Frame(window, bg='white', bd='0px')
    col_frame.grid(row=6, column=0)
    Col_label = tk.Label(col_frame,
                         text='Title column name of training data',
                         bg='white',
                         font=('Times New Roman', 10),
                         width='30',
                         anchor='e')
    Col_label.grid(row=0, column=0, padx=5)
    col_entry = tk.Entry(col_frame, width='25', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        col_entry.insert(0, default_str[3])
    col_entry.grid(row=0, column=1)

    ab_label = tk.Label(col_frame,
                        text='Abstract column name of training data',
                        bg='white',
                        font=('Times New Roman', 10),
                        width='30',
                        anchor='e')
    ab_label.grid(row=1, column=0, padx=5)
    ab_entry = tk.Entry(col_frame, width='25', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        ab_entry.insert(0, default_str[4])
    ab_entry.grid(row=1, column=1)
    entry_frame = tk.Frame(window, bg='white', bd='0px', heigh='20')
    entry_frame.grid(row=7, column=0)
    Label_label = tk.Label(entry_frame,
                           text='Column name of label data',
                           bg='white',
                           font=('Times New Roman', 10),
                           width='25',
                           anchor='e')
    Label_label.grid(row=0, column=0, padx=5)
    label_entry = tk.Entry(entry_frame, width='25', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        label_entry.insert(0, default_str[5])
    label_entry.grid(row=0, column=1, padx=5)

    # Execute AI
    execute_btn = tk.Button(entry_frame, text='Execute', anchor='nw', command=lambda: Threader(name='exe'))
    execute_btn.grid(row=1, column=0)

    msg_frame = tk.Frame(window, bg='white', bd='0px')
    msg_frame.grid(row=8, column=0, padx=10)

    msg_label = tk.Label(msg_frame, bg='white',
                         text='Process',
                         font=('Times New Roman', 10),
                         width='25',
                         anchor='w')
    msg_label.grid(row=0, column=0)
    msg_text = tk.Text(msg_frame,
                       bg='white',
                       font=('Times New Roman', 12),
                       height='7',
                       width='70',
                       padx='5')
    msg_text.grid(row=1, column=0)

    window.mainloop()