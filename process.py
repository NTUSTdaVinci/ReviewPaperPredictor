from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint

from ui.ui_process import Process_MainWindow


class Process_Window(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()  # in python3, super(Class, self).xxx = super().xxx
        self.ui = Process_MainWindow()
        self.ui.setupUi(self)
        self.setup_ui()

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
    #
    # def moveWindow(e):
    #
    #     if e.buttons() == Qt.LeftButton:
    #         self.move(self.pos())

    def mousePressEvent(self, event):
        self.oldPosition = event.globalPos()

    def mouseMoveEvent(self, event):
        delta = QPoint(event.globalPos() - self.oldPosition)
        self.move(self.x() + delta.x(), self.y() + delta.y())
        self.oldPosition = event.globalPos()

    def setup_ui(self):
        print("test")
        self.ui.closeButton.clicked.connect(self.close)
        self.ui.miniButton.clicked.connect(self.showMinimized)
        # self.ui.widget.mouseMoveEvent = self.moveWindow
        # self.ui.checkBox.stateChanged.connect(self.onCheckBoxClick)
        #
        # self.ui.closeButton.clicked.connect(self.close)
        # self.ui.miniButton.clicked.connect(self.showMinimized)
        #
        # self.ui.bu_input_button.clicked.connect(lambda: self.open_file(0))
        # self.ui.bu_wb_button.clicked.connect(lambda: self.open_file(1))
        # self.ui.bu_output_button.clicked.connect(lambda: self.open_folder(0))
        #
        # self.ui.pre_input_button.clicked.connect(lambda: self.open_file(2))
        # self.ui.pre_wb_button.clicked.connect(lambda: self.open_file(3))
        # self.ui.pre_model_button.clicked.connect(self.open_h5_file)
        # self.ui.pre_output_button.clicked.connect(lambda: self.open_folder(1))
        #
        # self.ui.bu_exeButton.clicked.connect(self.bu_check_file)
        # self.ui.pre_exeButton.clicked.connect(self.pre_check_file)

    def center(self):
        print("test")
        # qr = self.frameGeometry()
        # cp = QDesktopWidget().availableGeometry().center()
        # qr.moveCenter(cp)
        # self.move(qr.topLeft())

    def read_label_color_change(self):
        self.ui.label_read.setStyleSheet("color: #000;")
        self.ui.label_arr_check.clear()
        self.ui.label_arr_read.setPixmap(QtGui.QPixmap(".\\ui\\../assets/icon/arrow.png"))

    def data_label_change(self, num):
        self.ui.label_obtain.setStyleSheet("color: #000;")
        self.ui.label_obtain.setText("Obtain " + str(num) + " data")
        self.ui.label_arr_read.clear()
        self.ui.label_arr_ob.setPixmap(QtGui.QPixmap(".\\ui\\../assets/icon/arrow.png"))

    def keyword_label_color_change(self):
        self.ui.label_keyword.setStyleSheet("color: #000;")
        self.ui.label_arr_ob.clear()
        self.ui.label_arr_kw.setPixmap(QtGui.QPixmap(".\\ui\\../assets/icon/arrow.png"))

    def ai_label_color_change(self):
        self.ui.label_ai.setStyleSheet("color: #000;")
        self.ui.label_arr_kw.clear()
        self.ui.label_arr_ai.setPixmap(QtGui.QPixmap(".\\ui\\../assets/icon/arrow.png"))

    def saving_label_color_change(self):
        self.ui.label_saving.setStyleSheet("color: #000;")
        self.ui.label_arr_ai.clear()
        self.ui.label_arr_sa.setPixmap(QtGui.QPixmap(".\\ui\\../assets/icon/arrow.png"))

    def finished_label_color_change(self):
        self.ui.label_finish.setStyleSheet("color: #000;")
        self.ui.label_arr_sa.clear()
        self.ui.label_arr_fi.setPixmap(QtGui.QPixmap(".\\ui\\../assets/icon/arrow.png"))
