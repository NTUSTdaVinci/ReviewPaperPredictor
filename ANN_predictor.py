import os
import sys
import tkinter as tk
import threading
from tkinter import filedialog

from keras.models import load_model, Sequential

import numpy as np
import pandas as pd
import re

import time
import logging

TEXT_LINE_NO = 1

class Threader(threading.Thread):
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.daemon = True
        self.start()

    def run(self):
        file_check()


def file_check():
    """ Check all input files exist or not

        Get file and folder path from user interface
        Write all paths into default_value.txt
        Then check files exist or not

        Args:
        Returns:
        Raises:
    """

    print("Checking input files ...")
    msg_IO("Checking input files ...\n")

    file_path = input_entry.get()
    wb_path = wb_entry.get()
    model_path = model_entry.get()
    save_dir = output_entry.get()
    ti_name = col_entry.get()
    ab_name = ab_entry.get()

    # rewrite default file
    with open('default_value_pre.txt', 'w+') as f:
        f.write(file_path + "\n")
        f.write(wb_path + "\n")
        f.write(model_path + "\n")
        f.write(save_dir + "\n")
        f.write(ti_name + "\n")
        f.write(ab_name + "\n")

    # check file extension
    root_file, extension_file = os.path.splitext(file_path)
    root_wb, extension_wb = os.path.splitext(wb_path)
    root_md, extension_md = os.path.splitext(model_path)
    if extension_file == '.xlsx' and extension_wb == '.xlsx' and extension_md == '.h5':
        # check files or directory exist
        # if files exist, call build_ai() to train AI model
        if os.path.isfile(file_path) and os.path.isfile(wb_path) \
                and os.path.isfile(model_path) and os.path.isdir(save_dir):
            run_ai_model(file_path, model_path, wb_path, save_dir, ti_name, ab_name)
            msg_IO("Finished.")
        # if one of files doesn't exist, output error msg
        else:
            msg_IO("Files or directory do not exist.\n")
    else:
        msg_IO("Invalid files! Please check your input files.\n")

    sys.stdout.close()

def browseDataFile():
    tk.Tk().withdraw()

    input_entry.delete(0, 'end')
    input_entry.insert(0, filedialog.askopenfilename())


def browseWBFile():
    tk.Tk().withdraw()

    wb_entry.delete(0, 'end')
    wb_entry.insert(0, filedialog.askopenfilename())


def browseModelFile():
    tk.Tk().withdraw()

    model_entry.delete(0, 'end')
    model_entry.insert(0, filedialog.askopenfilename())


def browseFolder():
    tk.Tk().withdraw()

    output_entry.delete(0, 'end')
    output_entry.insert(0, filedialog.askdirectory())


def run_ai_model(file_path, model_path, wb_path, save_dir, ti_name, ab_name):
    """ Review paper AI Predictor

        Get input data from input file
        Calculate the number of keywords in title and abstract from every paper
        Turn keyword statistics table into input data of model
        Predict and store keyword statistics and predict result into output folder

        Args:
            file_path: path of training paper
            model_path: path of trained AI model
            wb_path: path of word bank
            save_dir: path of output folder
            ti_name: column name of title data
            ab_name: column name of abstract data

        Returns:
        Raises:
    """

    msg_IO('Reading data ...\n')

    # read word bank
    df_wb = pd.read_excel(wb_path)
    wbList = df_wb.values.tolist()
    print(wbList)

    # read training data from input file: title, abstract, label
    train = pd.read_excel(file_path)
    train[ti_name] = train[ti_name].astype(str)
    train[ab_name] = train[ab_name].astype(str)

    input = []
    input_counter = 0

    msg_IO("Obtain " + str(len(train[ti_name])) + " data\n")
    msg_IO("Calculating keyword in training data.\n")
    print("Calculating keyword in training data ...")
    # this loop travels every title and abstract of each paper for keywords calculation
    for ti_sentence, ab_sentence in zip(train[ti_name].values, train[ab_name].values):
        ti_list = []
        ab_list = []
        # calculate the number of each keyword in title and abstract
        # save the result in ti_list and ab_list
        for word_list in wbList:
            word_counter_ti = 0
            word_counter_ab = 0
            for word in word_list:
                if pd.isna(word):
                    break
                word_counter_ti += len(re.findall(word.lower(), str(ti_sentence).lower(), re.IGNORECASE))
                word_counter_ab += len(re.findall(word.lower(), str(ab_sentence).lower(), re.IGNORECASE))
            ti_list.append(word_counter_ti)
            ab_list.append(word_counter_ab)

        # combine the data stream from title and abstract
        sentence_list = ti_list + ab_list

        ti_list.clear()
        ab_list.clear()

        input.append(sentence_list)
        input_counter += 1

    input_data = np.array(input)

    model = Sequential()
    msg_IO("Loading model...\n")
    model = load_model(model_path)

    msg_IO("Predicting...\n")
    predict = model.predict(input_data)

    msg_IO("Saving output files...\n")

    col = np.append(df_wb['keyword'], df_wb['keyword'])
    print(len(col))
    df_word = pd.DataFrame(input, columns=col)
    df_word.to_excel(save_dir + "/keywords statistics.xlsx")

    col_name = train.columns.tolist()
    col_name.insert(0, 'Possibility of Review Paper')
    training_input = train.reindex(columns=col_name)
    training_input['Possibility of Review Paper'] = predict
    input_file_name = file_path.split('/')[-1].split('.')[0]
    output_file_name = input_file_name + '_' + str(int(time.time()))
    training_input.to_excel(save_dir + '/' + output_file_name + '.xlsx', index=False)

def msg_IO(msg):
    msg_text.insert('end', msg)
    print(msg)
    msg_text.update_idletasks()
    global TEXT_LINE_NO
    TEXT_LINE_NO += 1


if __name__ == '__main__':

    sys.stdout = open('log', 'w')
    # FORMAT = '%(asctime)s %(levelname)s: %(message)s'
    # logging.basicConfig(level=logging.DEBUG, filename='myLog.log', filemode='w', format=FORMAT)
    #
    # logging.debug('debug message')
    # logging.info('info message')
    # logging.warning('warning message')
    # logging.error('error message')
    # logging.critical('critical message')

    # Flag for default file:
    # True: exist, False: not exist
    DEFAULT_FILE = True
    # Check default file exist
    if os.path.isfile('default_value_pre.txt'):
        with open('default_value_pre.txt', 'r') as f:
            default_str = f.read().splitlines()
            print(default_str)

        # Check file format
        if len(default_str) != 6:
            DEFAULT_FILE = False
    else:
        DEFAULT_FILE = False

    # Draw UI
    window = tk.Tk()
    window.title('Review Paper Predictor')
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
                          text='1. Choose your input file paths by pressing "Browse file" button.\n'
                               '2. Enter column name of title and abstract data stream. \n'
                               '3. Press Execute button.\n'
                               '\nPLEASE CLOSE any output file the program generates before executing',
                          bg='white',
                          font=('Consolas', 12),
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
    input_entry = tk.Entry(bottom_frame, width='40', font=('Times New Roman', 12))

    # if default file exist, insert default value
    if DEFAULT_FILE:
        input_entry.insert(0, default_str[0])
    input_entry.grid(row=0, column=1)

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

    model_label = tk.Label(bottom_frame,
                           text='Model file path',
                           bg='white',
                           font=('Times New Roman', 10),
                           width='15',
                           anchor='nw')
    model_label.grid(row=2, column=0, sticky='E')
    model_entry = tk.Entry(bottom_frame, width='40', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        model_entry.insert(0, default_str[2])
    model_entry.grid(row=2, column=1)

    model_files_btn = tk.Button(bottom_frame, text='Browse file', anchor='nw', command=browseModelFile)
    model_files_btn.grid(row=2, column=2)

    # Output file
    output_label = tk.Label(bottom_frame,
                            text='Output folder path',
                            bg='white',
                            font=('Times New Roman', 10),
                            width='15',
                            anchor='nw')
    output_label.grid(row=3, column=0, sticky='E')
    output_entry = tk.Entry(bottom_frame, width='40', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        output_entry.insert(0, default_str[3])
    output_entry.grid(row=3, column=1)

    output_files_btn = tk.Button(bottom_frame, text='Browse folder', anchor='nw', command=browseFolder)
    output_files_btn.grid(row=3, column=2)

    col_frame = tk.Frame(window, bg='white', bd='0px')
    col_frame.grid(row=5, column=0)
    Col_label = tk.Label(col_frame,
                         text='Title column name of training data',
                         bg='white',
                         font=('Times New Roman', 10),
                         width='30',
                         anchor='e')
    Col_label.grid(row=0, column=0, padx=5)
    col_entry = tk.Entry(col_frame, width='25', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        col_entry.insert(0, default_str[4])
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
        ab_entry.insert(0, default_str[5])
    ab_entry.grid(row=1, column=1)
    entry_frame = tk.Frame(window, bg='white', bd='0px')
    entry_frame.grid(row=6, column=0)

    # Execute AI
    exe_frame = tk.Frame(window, bg='white', bd='0px', heigh='20')
    exe_frame.grid(row=8, column=0, padx=10)
    execute_btn = tk.Button(exe_frame, text='Execute',
                            anchor='nw', command=lambda: Threader(name='exe'))
    execute_btn.grid(row=0, column=0)

    msg_frame = tk.Frame(window, bg='white', bd='0px')
    msg_frame.grid(row=9, column=0, padx=10)

    msg_label = tk.Label(msg_frame, bg='white',
                         text='Process',
                         font=('Times New Roman', 10),
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
