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

from geometry import (
    D, build_affine_matrix, apply_affine_transform, WingSegment,
    recursive_sac_merge, get_transformed_contour, transformed_diagonal_sac,
    clip_polygon_to_boundary, add_nose_extension_triangle_with_vertical_leg,
    verticalize_polygon, vertical_sac_line_transformed, compute_total_area,
    compute_mac_for_wing, find_x25
)

class WingSACVisualizer(QMainWindow):
    """
    Основное окно визуализации САХ для многосегментных крыльев.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MACWING - Визуализатор САХ")
        self.setMinimumSize(900, 600)
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)

        splitter = QSplitter(QtCore.Qt.Horizontal)
        main_layout = QHBoxLayout()
        central.setLayout(main_layout)
        main_layout.addWidget(splitter)

        # Левая панель управления
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        self.tabWidget = QTabWidget()
        left_layout.addWidget(self.tabWidget, stretch=1)
        self.tabWidget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tabWidget.customContextMenuRequested.connect(self.wing_tab_context_menu)

        add_btn = QPushButton("Добавить новое крыло")
        add_btn.clicked.connect(self.add_wing_tab)
        left_layout.addWidget(add_btn)

        # Секция поворота
        rot_group = QWidget()
        rot_layout = QHBoxLayout()
        rot_group.setLayout(rot_layout)
        self.angle_edit = QLineEdit("15")
        self.pivot_x_edit = QLineEdit("1")
        self.pivot_y_edit = QLineEdit("0")
        rot_layout.addWidget(QLabel("Угол (°):"))
        rot_layout.addWidget(self.angle_edit)
        rot_layout.addWidget(QLabel("Pivot X:"))
        rot_layout.addWidget(self.pivot_x_edit)
        rot_layout.addWidget(QLabel("Pivot Y:"))
        rot_layout.addWidget(self.pivot_y_edit)
        left_layout.addWidget(rot_group)

        self.diag_checkbox = QCheckBox("Показывать диагонали")
        self.diag_checkbox.setChecked(True)
        left_layout.addWidget(self.diag_checkbox)

        # Кнопка для вызова окна «Утка»
        self.open_utka_btn = QPushButton("Открыть окно 'Утка'")
        self.open_utka_btn.clicked.connect(self.open_utka_window)
        left_layout.addWidget(self.open_utka_btn)

        left_layout.addStretch()

        # Правая панель с графиком
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        right_layout.addWidget(self.toolbar)
        right_layout.addWidget(self.canvas)

        right_layout.addWidget(QLabel("Информация по САХ:"))
        self.sac_info_text = QPlainTextEdit()
        self.sac_info_text.setReadOnly(True)
        right_layout.addWidget(self.sac_info_text)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])

        # Подключение сигналов
        self.angle_edit.textChanged.connect(self.update_plot)
        self.pivot_x_edit.textChanged.connect(self.update_plot)
        self.pivot_y_edit.textChanged.connect(self.update_plot)
        self.diag_checkbox.toggled.connect(self.update_plot)

        self.add_wing_tab()
        self.update_plot()

    def open_utka_window(self):
        self.utka_win = UtkaWindow()
        self.utka_win.show()

    def add_wing_tab(self):
        tbl = QTableWidget(0, 7)
        tbl.setHorizontalHeaderLabels([
            "Visible", "X Start", "Y Start",
            "Root Chord", "Tip Chord", "Sweep(°)", "Span"
        ])
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
        self.tabWidget.addTab(w, f"Крыло {idx}")
        tbl.cellChanged.connect(self.update_plot)

    def remove_wing_tab(self, index):
        if 0 <= index < self.tabWidget.count():
            self.tabWidget.removeTab(index)
        self.update_plot()

    def wing_tab_context_menu(self, pos):
        t_i = self.tabWidget.tabBar().tabAt(pos)
        menu = QMenu(self)
        add_act = menu.addAction("Добавить новое крыло")
        rm_act = None
        if t_i >= 0:
            rm_act = menu.addAction("Удалить текущее крыло")
        chosen = menu.exec_(self.tabWidget.mapToGlobal(pos))
        if chosen == add_act:
            self.add_wing_tab()
        elif chosen == rm_act:
            self.remove_wing_tab(t_i)

    def segments_context_menu(self, pos, table):
        menu = QMenu(self)
        add_act = menu.addAction("Добавить сегмент")
        rm_act = menu.addAction("Удалить сегмент")
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
        cly = QHBoxLayout(cw)
        cly.setContentsMargins(0, 0, 0, 0)
        cly.setAlignment(Qt.AlignCenter)
        cb = QCheckBox()
        cb.setChecked(bool(row_data[0]))
        cly.addWidget(cb)
        cw.setLayout(cly)
        table.setCellWidget(rw, 0, cw)
        cb.stateChanged.connect(self.update_plot)

        for col in range(1, 7):
            it = QTableWidgetItem(str(row_data[col]))
            it.setFlags(it.flags() | Qt.ItemIsEditable)
            table.setItem(rw, col, it)

    def _safe_float(self, text, default=0.0):
        try:
            return float(text.replace(',', '.'))
        except (ValueError, TypeError):
            return default

    def collect_segments_from_table(self, table):
        segs = []
        prev_x_tip = None
        prev_y_tip = None
        for r in range(table.rowCount()):
            cw = table.cellWidget(r, 0)
            if not cw: continue
            cbox = cw.findChild(QCheckBox)
            if not cbox or not cbox.isChecked(): continue
            
            try:
                # Явное чтение с проверкой
                vals = []
                for col in range(1, 7):
                    item = table.item(r, col)
                    if not item: raise ValueError
                    vals.append(float(item.text().replace(',', '.')))
                x_st, y_st, rc, tc, sw, sp = vals
            except ValueError:
                continue

            if not segs:
                s = WingSegment(rc, tc, sw, sp, x_st, y_st)
            else:
                s = WingSegment(rc, tc, sw, sp, prev_x_tip, prev_y_tip)
            
            segs.append(s)
            prev_x_tip, prev_y_tip = s.x_tip, s.y_tip
        return segs

    def update_plot(self):
        self.ax.clear()
        self.ax.invert_yaxis()

        angle_deg = self._safe_float(self.angle_edit.text(), 0.0)
        px = self._safe_float(self.pivot_x_edit.text(), 0.0)
        py = self._safe_float(self.pivot_y_edit.text(), 0.0)

        pivot = (D(px), D(py))
        angle_rad = math.radians(angle_deg)
        show_diag = self.diag_checkbox.isChecked()

        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        sac_info = []

        for w_i in range(self.tabWidget.count()):
            tw = self.tabWidget.widget(w_i)
            tbl = tw.findChild(QTableWidget)
            if not tbl: continue

            segs = self.collect_segments_from_table(tbl)
            if not segs: continue

            all_stages = recursive_sac_merge(segs)
            clr = color_list[w_i % len(color_list)]

            for stage_idx, seg_list in enumerate(all_stages, start=1):
                for seg_idx, segm in enumerate(seg_list, start=1):
                    M = build_affine_matrix(pivot, angle_rad)
                    shape_pts = [apply_affine_transform(pt, M) for pt in segm.get_contour()]

                    if len(shape_pts) >= 4:
                        xvals = [float(p[0]) for p in shape_pts]
                        yvals = [float(p[1]) for p in shape_pts]
                        xvals.append(xvals[0]); yvals.append(yvals[0])
                        lbl = f"Крыло {w_i+1}, этап {stage_idx}" if seg_idx == 1 else ""
                        self.ax.fill(xvals, yvals, alpha=0.3, edgecolor=clr, label=lbl)

                    if show_diag:
                        # Линии диагоналей
                        mod_pts = get_transformed_contour(segm, M)
                        mod_pts = clip_polygon_to_boundary(mod_pts, D("0"))
                        mod_pts = add_nose_extension_triangle_with_vertical_leg(mod_pts)
                        mod_pts = verticalize_polygon(mod_pts)
                        diag_res = transformed_diagonal_sac(mod_pts, D("1.0"))
                        if diag_res:
                            A_l, B_l, C_l, D_l, _ = diag_res
                            self.ax.plot([float(A_l[0]), float(D_l[0])], [float(A_l[1]), float(D_l[1])], 'k--', lw=1)
                            self.ax.plot([float(C_l[0]), float(B_l[0])], [float(C_l[1]), float(B_l[1])], 'k--', lw=1)

                    # SAC линия
                    sac_line = vertical_sac_line_transformed(segm, M, pivot, boundary=D("0"))
                    x_sac = sac_line[0][0]
                    y_lo, y_hi = sac_line[0][1], sac_line[1][1]
                    self.ax.plot([float(x_sac), float(x_sac)], [float(y_lo), float(y_hi)], 'r-', lw=2)
                    ym = (y_lo + y_hi) / D("2")
                    self.ax.plot(float(x_sac), float(ym), 'ro', ms=4)

                    L_sac = float(abs(y_hi - y_lo))
                    sac_info.append(f"Крыло {w_i+1}, этап {stage_idx}, сегм.{seg_idx}: L={L_sac:.6f}, center=({float(x_sac):.3f}, {float(ym):.3f})")
                    self.ax.text(float(x_sac), float(ym), f"SAC\nL={L_sac:.3f}", color='red', fontsize=8, ha='right')

        self.ax.plot(float(pivot[0]), float(pivot[1]), 'ks', ms=8, label="Pivot")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()
        self.sac_info_text.setPlainText("\n".join(sac_info))

class UtkaWindow(QMainWindow):
    """
    Окно для расчета конфигурации «утка».
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Расчет схемы «Утка»")
        self.setMinimumSize(900, 600)
        self.resize(1100, 750)

        central = QWidget()
        self.setCentralWidget(central)
        splitter = QSplitter(QtCore.Qt.Horizontal)
        main_layout = QHBoxLayout(central)
        main_layout.addWidget(splitter)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.tabWidget = QTabWidget()
        left_layout.addWidget(self.tabWidget, stretch=1)

        # Таблицы для переднего и заднего крыла
        self.front_table = self._create_wing_table()
        self.rear_table = self._create_wing_table()
        self.tabWidget.addTab(self.front_table, "Переднее крыло")
        self.tabWidget.addTab(self.rear_table, "Заднее крыло")

        # Параметры
        param_group = QWidget()
        param_layout = QVBoxLayout(param_group)
        
        row1 = QHBoxLayout()
        self.angle_edit = QLineEdit("0"); self.pivot_x_edit = QLineEdit("0.5"); self.pivot_y_edit = QLineEdit("0.0")
        row1.addWidget(QLabel("Угол:")); row1.addWidget(self.angle_edit)
        row1.addWidget(QLabel("Pivot X:")); row1.addWidget(self.pivot_x_edit)
        row1.addWidget(QLabel("Pivot Y:")); row1.addWidget(self.pivot_y_edit)
        param_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.delta_f_edit = QLineEdit("5"); self.n_edit = QLineEdit("0.2"); self.k_edit = QLineEdit("1.0")
        row2.addWidget(QLabel("Δφ:")); row2.addWidget(self.delta_f_edit)
        row2.addWidget(QLabel("N:")); row2.addWidget(self.n_edit)
        row2.addWidget(QLabel("K:")); row2.addWidget(self.k_edit)
        param_layout.addLayout(row2)

        self.diag_checkbox = QCheckBox("Диагонали"); self.diag_checkbox.setChecked(True)
        param_layout.addWidget(self.diag_checkbox)
        left_layout.addWidget(param_group)

        # График и результаты
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        self.figure, self.ax = plt.subplots(); self.canvas = FigureCanvas(self.figure)
        right_layout.addWidget(NavigationToolbar(self.canvas, self))
        right_layout.addWidget(self.canvas)
        self.result_text = QPlainTextEdit(); self.result_text.setReadOnly(True)
        right_layout.addWidget(self.result_text)

        splitter.addWidget(left_widget); splitter.addWidget(right_widget)
        splitter.setSizes([400, 700])

        # Коннекты
        for ed in [self.angle_edit, self.pivot_x_edit, self.pivot_y_edit, self.delta_f_edit, self.n_edit, self.k_edit]:
            ed.textChanged.connect(self.update_plot)
        self.diag_checkbox.toggled.connect(self.update_plot)
        
        self._set_defaults()
        self.update_plot()

    def _create_wing_table(self):
        tbl = QTableWidget(0, 7)
        tbl.setHorizontalHeaderLabels(["Visible", "X Start", "Y Start", "Root Chord", "Tip Chord", "Sweep(°)", "Span"])
        tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        tbl.cellChanged.connect(self.update_plot)
        return tbl

    def _set_defaults(self):
        def_f = [[True, 0.0, 0.3, 0.5, 0.3, 10, 0.4], [True, 0.0, 0.0, 0.4, 0.2, 15, 0.3]]
        def_r = [[True, 1.0, -0.3, 1.0, 0.4, 10, 1.5], [True, 0.0, 0.0, 0.5, 0.2, 20, 0.3]]
        for rd in def_f: self._add_row(self.front_table, rd)
        for rd in def_r: self._add_row(self.rear_table, rd)

    def _add_row(self, table, data):
        r = table.rowCount(); table.insertRow(r)
        cb = QCheckBox(); cb.setChecked(data[0]); cb.stateChanged.connect(self.update_plot)
        cw = QWidget(); ly = QHBoxLayout(cw); ly.setContentsMargins(0,0,0,0); ly.setAlignment(Qt.AlignCenter); ly.addWidget(cb)
        table.setCellWidget(r, 0, cw)
        for c in range(1, 7): table.setItem(r, c, QTableWidgetItem(str(data[c])))

    def _safe_float(self, text, default=0.0):
        try: return float(text.replace(',', '.'))
        except: return default

    def collect_segments(self, table):
        segs = []; px_t, py_t = None, None
        for r in range(table.rowCount()):
            cw = table.cellWidget(r, 0)
            if not cw or not cw.findChild(QCheckBox).isChecked(): continue
            try:
                vals = [float(table.item(r, c).text().replace(',', '.')) for c in range(1, 7)]
                x_s, y_s, rc, tc, sw, sp = vals
                s = WingSegment(rc, tc, sw, sp, x_s if not segs else px_t, y_s if not segs else py_t)
                segs.append(s); px_t, py_t = s.x_tip, s.y_tip
            except: continue
        return segs

    def update_plot(self):
        self.ax.clear(); self.ax.invert_yaxis(); self.ax.grid(True)
        
        ang = self._safe_float(self.angle_edit.text()); px = self._safe_float(self.pivot_x_edit.text()); py = self._safe_float(self.pivot_y_edit.text())
        df = self._safe_float(self.delta_f_edit.text()); n_v = self._safe_float(self.n_edit.text()); k_v = self._safe_float(self.k_edit.text(), 1.0)
        
        pivot = (D(px), D(py)); ang_rad = math.radians(ang); show_diag = self.diag_checkbox.isChecked()
        f_segs = self.collect_segments(self.front_table); r_segs = self.collect_segments(self.rear_table)
        
        info = []
        if f_segs: info.extend(self.plot_wing(f_segs, pivot, ang_rad, 'green', "Переднее", show_diag))
        if r_segs: info.extend(self.plot_wing(r_segs, pivot, ang_rad, 'blue', "Заднее", show_diag))
        
        if f_segs and r_segs:
            # Расчет утки
            s_p = compute_total_area(f_segs); s_3 = compute_total_area(r_segs)
            mac_f = compute_mac_for_wing(f_segs); mac_r = compute_mac_for_wing(r_segs)
            x25_f = find_x25(f_segs); x25_r = find_x25(r_segs)
            h = x25_r - x25_f
            
            if abs(h) > 1e-9:
                ratio = (s_3 / s_p) * k_v * (1 + 0.01 * df + 0.05 * n_v)
                x_e = x25_f + h / (1 + ratio)
                c_eq = (s_p * mac_f + s_3 * mac_r) / (s_p + s_3)
                x_le_eq = x_e - 0.25 * c_eq
                
                info.append(f"--- Утка ---")
                info.append(f"C_eq: {c_eq:.3f}, x_LE(eq): {x_le_eq:.3f}")
                info.append(f"Центровка 15..25%: {x_le_eq + 0.15*c_eq:.3f}..{x_le_eq + 0.25*c_eq:.3f}")
                
                # Отрисовка экв. хорды
                self._plot_eq_chord(f_segs + r_segs, x_le_eq, c_eq, ang_rad)

        self.ax.set_title(f"Утка: ang={ang}°, Δφ={df}, N={n_v}")
        self.ax.legend(); self.canvas.draw()
        self.result_text.setPlainText("\n".join(info))

    def plot_wing(self, segs, pivot, ang_rad, clr, label, show_diag):
        info = []
        stages = recursive_sac_merge(segs)
        for st_idx, stage in enumerate(stages, 1):
            for sg_idx, seg in enumerate(stage, 1):
                M = build_affine_matrix(pivot, ang_rad)
                pts = [apply_affine_transform(p, M) for p in seg.get_contour()]
                if len(pts) >= 4:
                    xv = [float(p[0]) for p in pts]; yv = [float(p[1]) for p in pts]
                    xv.append(xv[0]); yv.append(yv[0])
                    self.ax.fill(xv, yv, alpha=0.3, edgecolor=clr, label=f"{label} эт.{st_idx}" if sg_idx==1 else "")
                
                if show_diag:
                    m_pts = get_transformed_contour(seg, M)
                    m_pts = clip_polygon_to_boundary(m_pts, D("0"))
                    m_pts = add_nose_extension_triangle_with_vertical_leg(m_pts)
                    m_pts = verticalize_polygon(m_pts)
                    res = transformed_diagonal_sac(m_pts)
                    if res:
                        al, bl, cl, dl, _ = res
                        self.ax.plot([float(al[0]), float(dl[0])], [float(al[1]), float(dl[1])], 'k--', lw=1)
                        self.ax.plot([float(cl[0]), float(bl[0])], [float(cl[1]), float(bl[1])], 'k--', lw=1)

                sl = vertical_sac_line_transformed(seg, M, pivot)
                xs = sl[0][0]; ylo, yhi = sl[0][1], sl[1][1]
                self.ax.plot([float(xs), float(xs)], [float(ylo), float(yhi)], 'r-', lw=2)
                l_s = float(abs(yhi - ylo))
                info.append(f"{label} эт.{st_idx} сег.{sg_idx}: L={l_s:.4f}")
        return info

    def _plot_eq_chord(self, all_segs, x_le, c_eq, ang_rad):
        all_pts = []
        for s in all_segs:
            M = build_affine_matrix((D(0),D(0)), ang_rad)
            all_pts.extend([apply_affine_transform(p, M) for p in s.get_contour()])
        if not all_pts: return
        y_m = 0.5 * (float(min(p[1] for p in all_pts)) + float(max(p[1] for p in all_pts)))
        self.ax.plot([x_le, x_le + c_eq], [y_m, y_m], 'm-', lw=3, label="C_eq")
        self.ax.plot([x_le, x_le + c_eq], [y_m, y_m], 'mo', ms=6)
        self.ax.text(x_le + 0.5*c_eq, y_m, f"C_eq={c_eq:.3f}", color='magenta', ha='center', va='bottom')

