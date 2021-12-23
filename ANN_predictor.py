import os
import tkinter as tk
import threading

import logging


class Threader(threading.Thread):
    def __init__(self, *args, **kwargs):
        threading.Thread.__init__(self, *args, **kwargs)
        self.daemon = True
        self.start()

    def run(self):
        file_check()

def file_check():
    print("run AI")

def browseDataFile():
    print("run AI")

def browseWBFile():
    print("run AI")


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
                          text='Choose your input file path by pressing "Browse file" button.\n'
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

    model_label = tk.Label(bottom_frame,
                        text='Model file path',
                        bg='white',
                        font=('Times New Roman', 10),
                        width='15',
                        anchor='nw')
    model_label.grid(row=2, column=0, sticky='E')
    model_entry = tk.Entry(bottom_frame, width='40', font=('Times New Roman', 12))
    if DEFAULT_FILE:
        model_entry.insert(0, default_str[1])
    model_entry.grid(row=2, column=1)

    model_files_btn = tk.Button(bottom_frame, text='Browse file', anchor='nw', command=browseWBFile)
    model_files_btn.grid(row=2, column=2)

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
    entry_frame.grid(row=6, column=0)

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