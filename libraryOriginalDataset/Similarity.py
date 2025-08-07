import os
import sys
import glob
import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QListWidget,
    QTreeWidget, QTreeWidgetItem, QLabel, QSplitter
)
from PySide6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from pybaselines import Baseline

SG_WINDOW = 13
SG_POLY = 3
APPLY_NORMALIZATION = True
MY_X, MY_Y = None, None
SIMILARITY_CACHE = {}

def normalize(y):
    return (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)

def baseline_correction(y, lam=1e5, p=0.01):
    baseline, _ = Baseline().asls(y, lam=lam, p=p)
    return y - baseline

def load_csv_cm_file(file):
    try:
        df = pd.read_csv(file, skiprows=2)
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        mask = (x >= 500) & (x <= 1600)
        return x[mask], y[mask]
    except Exception as e:
        print(f"Error loading {file}: {e}")
        return None, None

def process_and_compare(file):
    x, y = load_csv_cm_file(file)
    if x is None or len(x) < 10:
        return None

    y = baseline_correction(y, lam=1e5, p=0.01)
    y = savgol_filter(y, SG_WINDOW, SG_POLY)
    if APPLY_NORMALIZATION:
        y = normalize(y)

    interp = interp1d(x, y, kind='linear', bounds_error=False, fill_value='extrapolate')
    y_interp = interp(MY_X)

    pearson = pearsonr(MY_Y, y_interp)[0] if np.std(MY_Y) and np.std(y_interp) else 0

    return {
        "file": file,
        "y_interp": y_interp,
        "pearson": pearson,
        "sfec": 0,
        "combined": pearson
    }

class RamanViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Raman Similarity Viewer")
        self.setGeometry(100, 100, 1200, 700)

        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        self.graph_widget = self.create_graph_widget()
        splitter.addWidget(self.graph_widget)

        self.right_panel = self.create_right_panel()
        splitter.addWidget(self.right_panel)
        splitter.setSizes([800, 400])

        self.populate_library()
        self.compute_top5()

    def create_graph_widget(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Raman Spectrum Comparison")
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity")
        self.ax.grid()
        self.ax.plot(MY_X, MY_Y, label="My Data", color="blue")
        self.ax.legend()
        return widget

    def create_right_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.top5_label = QLabel("Top 5")
        layout.addWidget(self.top5_label)
        self.similarity_list = QListWidget()
        self.similarity_list.setMaximumHeight(130)
        self.similarity_list.itemClicked.connect(self.on_top5_item_clicked)
        layout.addWidget(self.similarity_list)
        self.library_label = QLabel("Library")
        layout.addWidget(self.library_label)
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.itemClicked.connect(self.on_tree_item_clicked)
        layout.addWidget(self.tree)
        return panel

    def populate_library(self):
        root_path = os.getcwd()
        for root, dirs, files in os.walk(root_path):
            parent = QTreeWidgetItem([os.path.relpath(root, root_path)])
            added = False
            for file in sorted(files):
                if file.lower().endswith(".csv"):
                    full_path = os.path.join(root, file)
                    item = QTreeWidgetItem([os.path.splitext(file)[0]])
                    item.setData(0, Qt.UserRole, full_path)
                    parent.addChild(item)
                    added = True
            if added:
                self.tree.addTopLevelItem(parent)

    def compute_top5(self):
        global SIMILARITY_CACHE
        all_files = glob.glob("./**/*.csv", recursive=True)
        results = []
        for file in all_files:
            result = process_and_compare(file)
            if result:
                SIMILARITY_CACHE[file] = result
                results.append(result)
        top5 = sorted(results, key=lambda r: r["combined"], reverse=True)[:5]
        self.similarity_list.clear()
        for r in top5:
            name = os.path.splitext(os.path.basename(r["file"]))[0]
            self.similarity_list.addItem(f"{name} (C={r['combined']:.4f})")

    def on_tree_item_clicked(self, item, column):
        file_path = item.data(0, Qt.UserRole)
        if not file_path:
            return
        self.display_comparison(file_path)

    def on_top5_item_clicked(self, item):
        text = item.text()
        base_name = text.split(" (C=")[0]
        match = [fp for fp in SIMILARITY_CACHE if os.path.splitext(os.path.basename(fp))[0] == base_name]
        if match:
            self.display_comparison(match[0])

    def display_comparison(self, file_path):
        r = SIMILARITY_CACHE.get(file_path) or process_and_compare(file_path)
        if not r:
            return
        self.ax.cla()
        self.ax.set_title("Raman Spectrum Comparison")
        self.ax.set_xlabel("Wavenumber (cm⁻¹)")
        self.ax.set_ylabel("Intensity")
        self.ax.grid()
        self.ax.plot(MY_X, MY_Y, label="My Data", color="blue")
        self.ax.plot(MY_X, r["y_interp"], label=os.path.splitext(os.path.basename(r["file"]))[0], color="red")
        self.ax.legend()
        self.ax.text(0.02, 0.95, f"P={r['pearson']:.4f}", transform=self.ax.transAxes, fontsize=9)
        self.canvas.draw()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2

    def get_my_data(image_path):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError("TIFF 이미지 불러오기 실패")
        disp = ((img / img.max()) * 255).astype(np.uint8) if img.dtype == np.uint16 else img.copy()
        scale = min(1280 / disp.shape[1], 720 / disp.shape[0], 1.0)
        if scale < 1.0:
            disp = cv2.resize(disp, (int(disp.shape[1] * scale), int(disp.shape[0] * scale)))
        rows = []
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(rows) >= 2:
                    rows.clear()
                rows.append(int(y / scale))
        cv2.namedWindow("Select Rows")
        cv2.setMouseCallback("Select Rows", on_mouse)
        while len(rows) < 2:
            tmp = cv2.cvtColor(disp.copy(), cv2.COLOR_GRAY2BGR)
            for y in rows:
                y2 = int(y * scale)
                cv2.line(tmp, (0, y2), (tmp.shape[1], y2), (0, 255, 0), 2)
            cv2.imshow("Select Rows", tmp)
            if cv2.waitKey(20) == 27:
                sys.exit()
        cv2.destroyAllWindows()
        start, end = sorted(rows)
        intensity = np.sum(img[start:end+1, :], axis=0).astype(float)
        x_orig = np.linspace(950, 750, img.shape[1])
        plt.figure(figsize=(12,6))
        plt.plot(x_orig, intensity)
        plt.title("Click 2 peaks (nm)")
        peaks = plt.ginput(2, timeout=0)
        plt.close()
        idx = [np.abs(x_orig - pt[0]).argmin() for pt in peaks]
        target_nm = (852.1, 898.2)
        target_cm = (1001.4, 1602.3)
        nm_pts = x_orig[[idx[0], idx[1]]]
        m_nm = (target_nm[1] - target_nm[0]) / (nm_pts[1] - nm_pts[0])
        b_nm = target_nm[0] - m_nm * nm_pts[0]
        x_nm_adj = m_nm * x_orig + b_nm
        x_cm_tmp = 1e7 / x_nm_adj
        cm_pts = x_cm_tmp[[idx[0], idx[1]]]
        m_cm = (target_cm[1] - target_cm[0]) / (cm_pts[1] - cm_pts[0])
        b_cm = target_cm[0] - m_cm * cm_pts[0]
        x_cm = m_cm * x_cm_tmp + b_cm
        mask = (x_cm >= 500) & (x_cm <= 1600)
        x_final = x_cm[mask]
        y_final = intensity[mask]

        y_final = baseline_correction(y_final, lam=1e5, p=0.01)
        y_final = savgol_filter(y_final, SG_WINDOW, SG_POLY)
        if APPLY_NORMALIZATION:
            y_final = normalize(y_final)
        return x_final, y_final

    MY_X, MY_Y = get_my_data("original_Mono12_20250411_110738.tiff")

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    viewer = RamanViewer()
    viewer.show()
    sys.exit(app.exec())
