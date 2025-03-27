# graph_window.py
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

class GraphWindow(QMainWindow):
    def __init__(self, title="Graph Window"):
        super().__init__()
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 600, 400)

        self.canvas = FigureCanvas(Figure())
        self.ax = self.canvas.figure.subplots()

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central_widget)

        self.clicks = []  # 클릭한 x 좌표 저장용
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.peak_callback = None  # 외부에서 처리할 콜백 함수

    def update_plot(self, x_data, y_data, xlabel="X", reverse_x=False):
        self.ax.clear()
        self.ax.plot(x_data, y_data, color='red')
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel("Intensity")

        # Y축 최댓값
        y_max = max(y_data)
        self.ax.set_ylim(0, y_max)

        self.ax.set_xlim(max(x_data), min(x_data))

        self.canvas.draw()


    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        self.clicks.append(event.xdata)

        # 클릭 지점 시각적으로 표시 (X 마커)
        self.ax.plot(event.xdata, event.ydata, marker="x", color="black", markersize=5)
        self.canvas.draw()
        if len(self.clicks) == 2:
            if self.peak_callback:
                self.peak_callback(self.clicks)
            self.clicks = []
