import os
import sys 
import time
from PyQt6 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import pandas as pd
import numpy as np
import keras.models
from sklearn.neighbors import NearestNeighbors

class CustomPlotWidget(pg.PlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def wheelEvent(self, event):
        if event.modifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier:
            event.accept()
            delta = event.angleDelta().y()
            if delta < 0:
                self.setXRange(*[x + 0.05 for x in self.viewRange()[0]], padding=0)
            elif delta > 0:
                self.setXRange(*[x - 0.05 for x in self.viewRange()[0]], padding=0)
        else:
            super().wheelEvent(event)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(400, 530)
        self.centralwidget = QtWidgets.QWidget(parent=MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.graphWidget = CustomPlotWidget(parent=self.centralwidget)
        self.graphWidget.setObjectName("graphWidget")
        self.verticalLayout.addWidget(self.graphWidget)
        self.buttonBrowse = QtWidgets.QPushButton(parent=self.centralwidget)
        self.buttonBrowse.setObjectName("buttonBrowse")
        self.verticalLayout.addWidget(self.buttonBrowse)
        self.centralwidget.setLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.buttonBrowse.setText(_translate("MainWindow", "Open File"))

class ExampleApp(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.buttonBrowse.clicked.connect(self.VisualizeDataFile)
        self.graphWidget.setBackground('#ffffff')

        self.all_lines = []
        self.time_marks = []
        self.edge_lines = []
        self.right_line = None
        self.left_line = None
        self.SegmentLen = 200
        self.arr = np.zeros((0, self.SegmentLen))
        self.arr_nums = np.array([])
        self.graphWidget.scene().sigMouseClicked.connect(self.mouseClicked)
        self.graphWidget.installEventFilter(self)

    def VisualizeDataFile(self):
        a = time.time()
        options = QtWidgets.QFileDialog.Option.DontUseNativeDialog
        fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', '.', 'Text Files (*.txt);;Parquet Files (*.parquet)',
                                                            options=options)
        if fileName:
            if fileName.endswith('.parquet'):
                times, values, edges = self.ReadParquetFile(fileName)
            elif fileName.endswith('.txt'):
                times, values, edges = self.ReadTxtFile(fileName)
            else:
                print('Unlnown File Type')
                exit()
            self.series = self.graphWidget.plot(times, values)
            self.series.setPen(color='#000000', width=1)

            pen = pg.mkPen(color='r', width=1)

            print(len(self.all_lines))
            for line in self.all_lines:
                self.graphWidget.addItem(line)

        b = time.time()

        print(b - a)

    def ReadParquetFile(self, fileName):
        data = pd.read_parquet(fileName)
        self.PrepareData(data)
        return data['time'], data['value'], data['edge']

    def ReadTxtFile(self, fileName):
        with open(fileName, 'r') as file:
            lines = file.readlines()

        j = -1
        values = []
        times = []

        for line in lines:
            j += 1
            if j == 0:
                continue
            value = line[:10]
            times.append(float(j))
            values.append(float(value))

        df1 = pd.DataFrame(data=values, columns=['value'])
        df2 = pd.DataFrame(data=times, columns=['time'])
        df = pd.concat([df1, df2], axis=1)

        df['edge'] = 0

        mx = 0
        mx_ind = 0

        n = len(values)
        const = 175

        while (mx_ind + const < n):
            mx = max(values[mx_ind + 1:mx_ind + const + 1])

            for j in range(mx_ind + const, mx_ind, -1):
                if values[j] == mx:
                    mx_ind = j
                    df['edge'].at[j] = 1
                    break
        fileName = fileName.replace('.txt', '.parquet')
        df.to_parquet(fileName)

        return self.ReadParquetFile(fileName)

    def PrepareData(self, data):
        Length = data.shape[0] - data.shape[0] % 200
        tmp = data['value']
        maxx = np.max(tmp)
        minn = np.min(tmp)
        tmp = (tmp - minn) / (maxx - minn)
        mx = 0
        for i in range(0, self.SegmentLen):
            if (data['edge'].at[i] == 1):
                mx = i
                break

        count = 10000
        pen = pg.mkPen(color='r', width=1)

        while (mx + self.SegmentLen < Length + 1):
            arr2 = np.zeros(self.SegmentLen)
            for i in range(mx + 1, mx + self.SegmentLen):
                if data['edge'].at[i] == 1:
                    line = pg.InfiniteLine(i, pen=pen)
                    self.all_lines.append(line)
                    self.edge_lines.append(i)

                    if i - mx > 10 and i + 1 - mx <= self.SegmentLen:
                        arr2[:i + 1 - mx] = tmp[mx:i + 1]
                        arr2[i + 1 - mx:] = 0
                        self.arr = np.vstack((self.arr, arr2))
                        self.arr_nums = np.append(self.arr_nums, i)
                    mx = i
                    break
            if mx > count:
                print(mx)
                count += 10000

    def mouseClicked(self, event):
        pos = event.scenePos()
        if self.graphWidget.plotItem.sceneBoundingRect().contains(pos):
            mousePoint = self.graphWidget.plotItem.vb.mapSceneToView(pos)
            pen = pg.mkPen(color='b', width=1)
            for i in range(len(self.edge_lines) - 1):
                if self.edge_lines[i] < mousePoint.x() < self.edge_lines[i + 1]:
                    if self.right_line is not None and self.left_line is not None:
                        self.graphWidget.removeItem(self.right_line)
                        self.graphWidget.removeItem(self.left_line)
                    self.right_line = pg.InfiniteLine(self.edge_lines[i], pen=pen)
                    self.left_line = pg.InfiniteLine(self.edge_lines[i + 1], pen=pen)
                    self.graphWidget.addItem(self.right_line)
                    self.graphWidget.addItem(self.left_line)

    def eventFilter(self, source, event):
        if (event.type() == QtCore.QEvent.Type.KeyPress and source is self.graphWidget):
            if event.key() == QtCore.Qt.Key.Key_Return:
                if self.right_line is not None and self.left_line is not None:
                    self.FindSimilarPart()
        return super(ExampleApp, self).eventFilter(source, event)

    def FindSimilarPart(self):
        index = np.where(self.arr_nums == self.left_line.getPos()[0])
        find_arr = self.arr[index[0][0]][:]

        if len(find_arr) < self.SegmentLen:
            i = len(find_arr)
            while i != self.SegmentLen:
                find_arr = np.append(find_arr, 0.0)
                i += 1

        encoder = keras.models.load_model("ecg_dense_encoder_bad_all_3.h5")
        self.arr = np.reshape(self.arr, (self.arr.shape[0], self.arr.shape[1], 1))
        pred = encoder.predict(self.arr)

        nei = NearestNeighbors(metric="euclidean")
        nei.fit(pred)

        NumOfNeighbors = 30

        def get_similar(find_arr, n_neighbors=NumOfNeighbors):
            encoded_find_arr = encoder.predict(find_arr)
            (distances,), (idx,) = nei.kneighbors(encoded_find_arr, n_neighbors=n_neighbors)
            return distances, self.arr[idx], idx

        temp = np.reshape(find_arr, (1, len(find_arr), 1))
        distances, neighbors, nums = get_similar(temp)

        for i in range(1, NumOfNeighbors):
            pen = pg.mkPen(color='g', width=1)
            line1 = pg.InfiniteLine(self.arr_nums[nums[i]], pen=pen)
            line2 = pg.InfiniteLine(self.arr_nums[nums[i] - 1], pen=pen)
            self.graphWidget.addItem(line1)
            self.graphWidget.addItem(line2)

    def wheelEvent(self, event):
        pos = event.scenePos()
        if self.graphWidget.plotItem.sceneBoundingRect().contains(pos):
            if event.modifiers() == QtCore.Qt.KeyboardModifier.ShiftModifier:
                delta = event.angleDelta().y()
                factor = 0.1
                if delta > 0:
                    self.graphWidget.setXRange(self.graphWidget.viewRange()[0][0] - factor,
                                               self.graphWidget.viewRange()[0][1] - factor)
                else:
                    self.graphWidget.setXRange(self.graphWidget.viewRange()[0][0] + factor,
                                               self.graphWidget.viewRange()[0][1] + factor)
            else:
                QtWidgets.QMainWindow.wheelEvent(self, event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = ExampleApp() 
    window.show() 
    app.exec()

if __name__ == '__main__':
    main()
