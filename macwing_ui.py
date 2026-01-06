import sys
import math
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QPlainTextEdit, QSplitter, QCheckBox, QTabWidget, QMenu, QPushButton
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from i18n import I18n
from geometry import (
    D, build_affine_matrix, apply_affine_transform, WingSegment,
    recursive_sac_merge, get_transformed_contour, transformed_diagonal_sac,
    clip_polygon_to_boundary, add_nose_extension_triangle_with_vertical_leg,
    verticalize_polygon, vertical_sac_line_transformed
)

class WingSACVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)

        splitter = QSplitter(QtCore.Qt.Horizontal)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        main_layout.addWidget(splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        # Переключатель языка
        self.lang_btn = QPushButton()
        self.lang_btn.clicked.connect(self.toggle_language)
        left_layout.addWidget(self.lang_btn)

        self.tabWidget = QTabWidget()
        left_layout.addWidget(self.tabWidget, stretch=1)
        self.tabWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tabWidget.customContextMenuRequested.connect(self.wing_tab_context_menu)

        self.add_btn = QPushButton()
        self.add_btn.clicked.connect(self.add_wing_tab)
        left_layout.addWidget(self.add_btn)

        rot_group = QWidget()
        rot_layout = QHBoxLayout()
        rot_group.setLayout(rot_layout)
        self.angle_edit = QLineEdit("15")
        self.pivot_x_edit = QLineEdit("1")
        self.pivot_y_edit = QLineEdit("0")
        
        self.lbl_angle = QLabel()
        self.lbl_px = QLabel()
        self.lbl_py = QLabel()
        
        rot_layout.addWidget(self.lbl_angle)
        rot_layout.addWidget(self.angle_edit)
        rot_layout.addWidget(self.lbl_px)
        rot_layout.addWidget(self.pivot_x_edit)
        rot_layout.addWidget(self.lbl_py)
        rot_layout.addWidget(self.pivot_y_edit)
        left_layout.addWidget(rot_group)

        self.diag_checkbox = QCheckBox()
        self.diag_checkbox.setChecked(True)
        left_layout.addWidget(self.diag_checkbox)

        left_layout.addStretch()

        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        self.lbl_info = QLabel()
        right_layout.addWidget(self.lbl_info)
        self.sac_info_text = QPlainTextEdit()
        self.sac_info_text.setReadOnly(True)
        right_layout.addWidget(self.sac_info_text)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])

        self.angle_edit.textChanged.connect(self.update_plot)
        self.pivot_x_edit.textChanged.connect(self.update_plot)
        self.pivot_y_edit.textChanged.connect(self.update_plot)
        self.diag_checkbox.toggled.connect(self.update_plot)

        self.retranslate_ui()
        self.add_wing_tab()
        self.update_plot()

    def toggle_language(self):
        I18n.toggle_lang()
        self.retranslate_ui()
        self.update_plot()

    def retranslate_ui(self):
        self.setWindowTitle(I18n.t("app_title"))
        self.lang_btn.setText(I18n.t("lang_toggle"))
        self.add_btn.setText(I18n.t("add_wing"))
        self.lbl_angle.setText(I18n.t("angle"))
        self.lbl_px.setText(I18n.t("pivot_x"))
        self.lbl_py.setText(I18n.t("pivot_y"))
        self.diag_checkbox.setText(I18n.t("show_diagonals"))
        self.lbl_info.setText(I18n.t("mac_info"))
        
        for i in range(self.tabWidget.count()):
            tbl = self.tabWidget.widget(i).findChild(QTableWidget)
            if tbl:
                tbl.setHorizontalHeaderLabels(I18n.t("table_headers"))
            self.tabWidget.setTabText(i, f"{I18n.t('wing_tab')} {i+1}")

    def add_wing_tab(self):
        tbl = QTableWidget(0, 7)
        tbl.setHorizontalHeaderLabels(I18n.t("table_headers"))
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.setContextMenuPolicy(Qt.CustomContextMenu)
        tbl.customContextMenuRequested.connect(lambda pos, t=tbl: self.segments_context_menu(pos, t))

        defaults = [
            [True, 0.0, 0.0, 1.0, 0.4, 10, 1.5],
            [True, 0.0, 0.0, 0.4, 0.2, 20, 0.3],
            [True, 0.0, 0.0, 0.2, 0.05, 22, 0.2],
        ]
        for rd in defaults:
            self.add_segment_row(tbl, rd)

        w = QWidget()
        ly = QVBoxLayout()
        ly.addWidget(tbl)
        w.setLayout(ly)
        idx = self.tabWidget.count() + 1
        self.tabWidget.addTab(w, f"{I18n.t('wing_tab')} {idx}")
        tbl.cellChanged.connect(self.update_plot)

    def wing_tab_context_menu(self, pos):
        t_i = self.tabWidget.tabBar().tabAt(pos)
        menu = QMenu(self)
        add_act = menu.addAction(I18n.t("add_wing_menu"))
        rm_act = None
        if t_i >= 0:
            rm_act = menu.addAction(I18n.t("remove_wing_menu"))
        chosen = menu.exec_(self.tabWidget.mapToGlobal(pos))
        if chosen == add_act:
            self.add_wing_tab()
        elif chosen == rm_act:
            if 0 <= t_i < self.tabWidget.count():
                self.tabWidget.removeTab(t_i)
                self.update_plot()

    def segments_context_menu(self, pos, table):
        menu = QMenu(self)
        add_act = menu.addAction(I18n.t("add_segment_menu"))
        rm_act = menu.addAction(I18n.t("remove_segment_menu"))
        r = table.rowAt(pos.y())
        chosen = menu.exec_(table.mapToGlobal(pos))
        if chosen == add_act:
            ins = r + 1 if r >= 0 else table.rowCount()
            self.add_segment_row(table, [True, 0.0, 0.0, 1.0, 0.4, 10, 1.0], ins)
        elif chosen == rm_act:
            if 0 <= r < table.rowCount():
                table.removeRow(r)
                self.update_plot()

    def add_segment_row(self, table, row_data, insert_at=None):
        rw = insert_at if insert_at is not None else table.rowCount()
        table.insertRow(rw)
        cw = QWidget()
        cly = QHBoxLayout(cw); cly.setContentsMargins(0, 0, 0, 0); cly.setAlignment(Qt.AlignCenter)
        cb = QCheckBox(); cb.setChecked(bool(row_data[0])); cly.addWidget(cb)
        table.setCellWidget(rw, 0, cw)
        cb.stateChanged.connect(self.update_plot)
        for col in range(1, 7):
            it = QTableWidgetItem(str(row_data[col]))
            table.setItem(rw, col, it)

    def _safe_float(self, text, default=0.0):
        try: return float(text.replace(',', '.'))
        except: return default

    def collect_segments_from_table(self, table):
        segs = []; px_t, py_t = None, None
        for r in range(table.rowCount()):
            cw = table.cellWidget(r, 0)
            if not cw or not cw.findChild(QCheckBox).isChecked(): continue
            try:
                vals = [float(table.item(r, c).text().replace(',', '.')) for c in range(1, 7)]
                x_st, y_st, rc, tc, sw, sp = vals
                if not segs: s = WingSegment(rc, tc, sw, sp, x_st, y_st)
                else: s = WingSegment(rc, tc, sw, sp, px_t, py_t)
                segs.append(s); px_t, py_t = s.x_tip, s.y_tip
            except: continue
        return segs

    def update_plot(self):
        self.ax.clear(); self.ax.invert_yaxis()
        ang = self._safe_float(self.angle_edit.text()); px = self._safe_float(self.pivot_x_edit.text()); py = self._safe_float(self.pivot_y_edit.text())
        pivot = (D(px), D(py)); ang_rad = math.radians(ang); show_diag = self.diag_checkbox.isChecked()
        sac_info = []
        for w_i in range(self.tabWidget.count()):
            tbl = self.tabWidget.widget(w_i).findChild(QTableWidget)
            segs = self.collect_segments_from_table(tbl)
            if not segs: continue
            stages = recursive_sac_merge(segs); clr = ['b', 'g', 'r', 'c', 'm', 'y', 'k'][w_i % 7]
            for st_idx, stage in enumerate(stages, 1):
                for sg_idx, seg in enumerate(stage, 1):
                    M = build_affine_matrix(pivot, ang_rad)
                    pts = [apply_affine_transform(p, M) for p in seg.get_contour()]
                    if len(pts) >= 4:
                        xv = [float(p[0]) for p in pts]; yv = [float(p[1]) for p in pts]
                        xv.append(xv[0]); yv.append(yv[0])
                        self.ax.fill(xv, yv, alpha=0.3, edgecolor=clr, label=I18n.t("plot_wing_label").format(w=w_i+1, s=st_idx) if sg_idx==1 else "")
                    if show_diag:
                        m_pts = verticalize_polygon(add_nose_extension_triangle_with_vertical_leg(clip_polygon_to_boundary(get_transformed_contour(seg, M), D("0"))))
                        res = transformed_diagonal_sac(m_pts)
                        if res:
                            al, bl, cl, dl, _ = res
                            self.ax.plot([float(al[0]), float(dl[0])], [float(al[1]), float(dl[1])], 'k--', lw=1)
                            self.ax.plot([float(cl[0]), float(bl[0])], [float(cl[1]), float(bl[1])], 'k--', lw=1)
                    sl = vertical_sac_line_transformed(seg, M, pivot)
                    xs, ylo, yhi = sl[0][0], sl[0][1], sl[1][1]; ym = (ylo + yhi) / D("2")
                    self.ax.plot([float(xs), float(xs)], [float(ylo), float(yhi)], 'r-', lw=2); self.ax.plot(float(xs), float(ym), 'ro', ms=4)
                    L_sac = float(abs(yhi - ylo))
                    sac_info.append(I18n.t("sac_info_format").format(w=w_i+1, s=st_idx, seg=sg_idx, L=L_sac, x=float(xs), y=float(ym)))
                    self.ax.text(float(xs), float(ym), f"SAC\nL={L_sac:.3f}", color='red', fontsize=8, ha='right')
        self.ax.plot(float(pivot[0]), float(pivot[1]), 'ks', ms=8, label="Pivot")
        self.ax.set_xlabel("X"); self.ax.set_ylabel("Y"); self.ax.grid(True); self.ax.legend(); self.canvas.draw()
        self.sac_info_text.setPlainText("\n".join(sac_info))
