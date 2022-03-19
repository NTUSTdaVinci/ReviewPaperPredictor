from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QDesktopWidget
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import os, time
import numpy as np
import pandas as pd

from ui.ui_main import Ui_MainWindow
from ann.builder import BuildingTask
from ann.predictor import PredictionTask

from ann.keyword_calculator import KeywordCalculator

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_ui()
        self.set_default_bu_value()
        self.set_default_pre_value()

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

    def setup_ui(self):
        self.ui.checkBox.stateChanged.connect(self.onCheckBoxClick)

        self.ui.closeButton.clicked.connect(self.close)
        self.ui.miniButton.clicked.connect(self.showMinimized)

        self.ui.bu_input_button.clicked.connect(lambda: self.open_file(0))
        self.ui.bu_wb_button.clicked.connect(lambda: self.open_file(1))
        self.ui.bu_output_button.clicked.connect(lambda: self.open_folder(0))

        self.ui.pre_input_button.clicked.connect(lambda: self.open_file(2))
        self.ui.pre_wb_button.clicked.connect(lambda: self.open_file(3))
        self.ui.pre_model_button.clicked.connect(self.open_h5_file)
        self.ui.pre_output_button.clicked.connect(lambda: self.open_folder(1))

        self.ui.bu_exeButton.clicked.connect(self.bu_check_file)
        self.ui.pre_exeButton.clicked.connect(self.pre_check_file)

    def bu_check_file(self):
        input_path = self.ui.bu_input_path.text()
        wb_path = self.ui.bu_wb_path.text()
        output_path = self.ui.bu_output_path.text()
        ti_col = self.ui.bu_ti_col.text()
        ab_col = self.ui.bu_ab_col.text()
        label_col = self.ui.bu_label_col.text()
        # rewrite default file
        with open('default_bu_value.txt', 'w+') as f:
            f.write(input_path + "\n")
            f.write(wb_path + "\n")
            f.write(output_path + "\n")
            f.write(ti_col + "\n")
            f.write(ab_col + "\n")
            f.write(label_col)

        # check file extension
        root_file, extension_file = os.path.splitext(input_path)
        root_wb, extension_wb = os.path.splitext(wb_path)
        if extension_file == '.xlsx' and extension_wb == '.xlsx':
            # check files or directory exist
            # if files exist, call build_ai() to train AI model
            if os.path.isfile(input_path)\
                    and os.path.isfile(wb_path)\
                    and os.path.isdir(output_path):
                self.bu_qthread = BuildingTask(input_path, wb_path, output_path,
                                               ti_col, ab_col, label_col)
                self.bu_qthread.qthread_signal.connect(self.progress_changed)
                self.bu_qthread.finished.connect(self.bu_finished)
                self.bu_qthread.start()
                self.ui.bu_exeButton.setEnabled(False)
                self.ui.checkBox.setEnabled(False)
            # if one of files doesn't exist, output error msg
            else:
                print("Files or directory do not exist.")
        else:
            print("Invalid files! Please check your input files.\n")

    def pre_check_file(self):
        input_path = self.ui.pre_input_path.text()
        wb_path = self.ui.pre_wb_path.text()
        model_path = self.ui.pre_model_path.text()
        output_path = self.ui.bu_output_path.text()
        ti_col = self.ui.bu_ti_col.text()
        ab_col = self.ui.bu_ab_col.text()

        # rewrite default file
        with open('default_pre_value.txt', 'w+') as f:
            f.write(input_path + "\n")
            f.write(wb_path + "\n")
            f.write(model_path + "\n")
            f.write(output_path + "\n")
            f.write(ti_col + "\n")
            f.write(ab_col + "\n")

        # check file extension
        root_file, extension_file = os.path.splitext(input_path)
        root_wb, extension_wb = os.path.splitext(wb_path)
        root_md, extension_md = os.path.splitext(model_path)
        if extension_file == '.xlsx' and extension_wb == '.xlsx' and extension_md == '.h5':
            # check files or directory exist
            # if files exist, call build_ai() to train AI model
            if os.path.isfile(input_path)\
                    and os.path.isfile(wb_path) \
                    and os.path.isfile(model_path) \
                    and os.path.isdir(output_path):
                self.pre_qthread = PredictionTask(input_path, wb_path, model_path,
                                                  output_path, ti_col, ab_col)
                self.pre_qthread.qthread_signal.connect(self.progress_changed)
                self.pre_qthread.finished.connect(self.pre_finished)
                self.pre_qthread.start()
                self.ui.pre_exeButton.setEnabled(False)
                self.ui.checkBox.setEnabled(False)
            # if one of files doesn't exist, output error msg
            else:
                print("Files or directory do not exist.")
        else:
            print("Invalid files! Please check your input files.\n")

    def progress_changed(self, text):
        print(text)

    def bu_finished(self):
        self.ui.bu_exeButton.setEnabled(True)
        self.ui.checkBox.setEnabled(True)

    def pre_finished(self):
        self.ui.pre_exeButton.setEnabled(True)
        self.ui.checkBox.setEnabled(True)

    def set_default_bu_value(self):
        # Check config file exist
        if os.path.isfile('default_bu_value.txt'):
            with open('default_bu_value.txt', 'r') as f:
                default_str = f.read().splitlines()
                print(default_str)

            # Check file format
            if len(default_str) == 6:
                self.ui.bu_input_path.setText(default_str[0])
                self.ui.bu_wb_path.setText(default_str[1])
                self.ui.bu_output_path.setText(default_str[2])
                self.ui.bu_ti_col.setText(default_str[3])
                self.ui.bu_ab_col.setText(default_str[4])
                self.ui.bu_label_col.setText(default_str[5])

    def set_default_pre_value(self):
        # Check config file exist
        if os.path.isfile('default_pre_value.txt'):
            with open('default_pre_value.txt', 'r') as f:
                default_str = f.read().splitlines()
                print(default_str)

            # Check file format
            if len(default_str) == 6:
                self.ui.pre_input_path.setText(default_str[0])
                self.ui.pre_wb_path.setText(default_str[1])
                self.ui.pre_model_path.setText(default_str[2])
                self.ui.pre_output_path.setText(default_str[3])
                self.ui.pre_ti_col.setText(default_str[4])
                self.ui.pre_ab_col.setText(default_str[5])

    def onCheckBoxClick(self):
        if self.ui.checkBox.isChecked():
            self.ui.stackedWidget_main.setCurrentIndex(1)
            self.ui.stackedWidget_input.setCurrentIndex(1)
            self.ui.menu.setStyleSheet("#menu{\n"
                                       "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:0.5, x3:0, y3:1 stop:0 #EBBDA2 , stop:1 #89D2DC, stop:2 #89D2DC);\n"
                                       " border-radius: 10px;\n"
                                       " border: none;\n"
                                       "}\n"
                                       "")
        else:
            self.ui.stackedWidget_main.setCurrentIndex(0)
            self.ui.stackedWidget_input.setCurrentIndex(0)
            self.ui.menu.setStyleSheet("#menu{\n"
                                       "background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:0.5, x3:0, y3:1 stop:0 #AABCDB , stop:1 #9CC4B2, stop:2 #9CC4B2);\n"
                                       " border-radius: 10px;\n"
                                       " border: none;\n"
                                       "}\n"
                                       "")

    def open_file(self, label_type):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "Open file",
                                                         "./",
                                                         'Excel Files (*.xlsx)')  # start path
        print(filename, filetype)
        if label_type == 0:
            self.ui.bu_input_path.setText(filename)
        elif label_type == 1:
            self.ui.bu_wb_path.setText(filename)
        elif label_type == 2:
            self.ui.pre_input_path.setText(filename)
        elif label_type == 3:
            self.ui.pre_wb_path.setText(filename)

    def open_h5_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                                                         "Open file",
                                                         "./",
                                                         'H5 Files (*.h5)')  # start path
        self.ui.pre_model_path.setText(filename)

    def open_folder(self, label_type):
        folder_path = QFileDialog.getExistingDirectory(self,
                                                       "Open folder",
                                                       "./")  # start path
        print(folder_path)
        if label_type == 0:
            self.ui.bu_output_path.setText(folder_path)
        elif label_type == 1:
            self.ui.pre_output_path.setText(folder_path)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

