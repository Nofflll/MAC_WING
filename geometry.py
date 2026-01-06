from decimal import Decimal, getcontext
import math

# Настройка точности Decimal
getcontext().prec = 15

def D(x):
    """Преобразование в Decimal для точности."""
    return Decimal(str(x))

def line_equation(p1, p2, tol=None):
    if tol is None:
        tol = D("1e-12")
    x1, y1 = D(p1[0]), D(p1[1])
    x2, y2 = D(p2[0]), D(p2[1])
    dx = x2 - x1
    if abs(dx) < tol:
        # Вертикальная прямая
        return None, x1
    m = (y2 - y1) / dx
    b = y1 - m * x1
    return m, b

def line_intersection(m1, b1, m2, b2, tol=None):
    if tol is None:
        tol = D("1e-12")
    if m1 is None and m2 is None:
        return None
    if m1 is None:
        x = b1
        y = m2 * x + b2 if m2 is not None else None
        return (x, y)
    if m2 is None:
        x = b2
        y = m1 * x + b1
        return (x, y)
    if abs(m1 - m2) < tol:
        return None
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return (x, y)

def refine_intersection(m1, b1, m2, b2, sac, iterations=10, tol=None):
    if tol is None:
        tol = D("1e-12")
    x, y = sac
    for _ in range(iterations):
        f1 = m1 * x + b1
        f2 = m2 * x + b2
        err = f1 - f2
        if abs(m1 - m2) > tol:
            x = x - err / (m1 - m2)
        else:
            break
    y = (m1 * x + b1 + m2 * x + b2) / D("2")
    return (x, y)

def build_affine_matrix(pivot, angle_rad):
    """Матрица поворота вокруг pivot: T(pivot)*R(angle)*T(-pivot)."""
    cos_a = D(math.cos(angle_rad))
    sin_a = D(math.sin(angle_rad))
    px, py = D(pivot[0]), D(pivot[1])
    tx = px - cos_a * px + sin_a * py
    ty = py - sin_a * px - cos_a * py
    return [
        [cos_a, -sin_a, tx],
        [sin_a,  cos_a, ty],
        [D("0"), D("0"), D("1")]
    ]

def apply_affine_transform(pt, M):
    x, y = D(pt[0]), D(pt[1])
    x_new = M[0][0] * x + M[0][1] * y + M[0][2]
    y_new = M[1][0] * x + M[1][1] * y + M[1][2]
    return (x_new, y_new)

def clip_polygon_to_boundary(poly, boundary, tol=None):
    """Обрезка многоугольника по вертикальной границе x >= boundary."""
    if tol is None:
        tol = D("1e-12")
    def inside(p):
        return p[0] >= boundary

    def intersect(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        if abs(x2 - x1) < tol:
            return (D(boundary), (y1 + y2) / D("2"))
        t = (D(boundary) - x1) / (x2 - x1)
        y_ = y1 + t * (y2 - y1)
        return (D(boundary), y_)

    if not poly:
        return []
    out = []
    prev = poly[-1]
    for cur in poly:
        if inside(cur):
            if not inside(prev):
                out.append(intersect(prev, cur))
            out.append(cur)
        elif inside(prev):
            out.append(intersect(prev, cur))
        prev = cur
    return out

def add_nose_extension_triangle_with_vertical_leg(poly, tol=None):
    """Добавляет «нос» к многоугольнику, формируя вертикальную сторону."""
    if tol is None:
        tol = D("1e-6")
    n = len(poly)
    if n < 2:
        return poly
    max_avg = -Decimal("Infinity")
    nose_i = 0
    for i in range(n):
        p1 = poly[i]
        p2 = poly[(i+1) % n]
        avg = (p1[0] + p2[0]) / D("2")
        if avg > max_avg:
            max_avg = avg
            nose_i = i
    A = poly[nose_i]
    B = poly[(nose_i+1) % n]
    if A[1] >= B[1]:
        U, L = A, B
        U_i = nose_i
    else:
        U, L = B, A
        U_i = (nose_i+1) % n
    R_i = (U_i - 1) % n
    R = poly[R_i]
    mR, bR = line_equation(R, U)
    vx = L[0]
    if mR is None:
        N = (vx, U[1])
    else:
        N = (vx, mR*vx + bR)
    newp = poly[:]
    ins = U_i + 1
    if ins > len(newp):
        ins = len(newp)
    newp.insert(ins, N)
    return newp

def verticalize_polygon(poly, tol=None):
    """«Вертикализует» многоугольник, прижимая точки к min/max X."""
    if tol is None:
        tol = D("1e-6")
    if not poly:
        return poly
    xs = [p[0] for p in poly]
    mn = min(xs)
    mx = max(xs)
    out = []
    n = len(poly)
    for i in range(n):
        c = list(poly[i])
        nxt = poly[(i+1) % n]
        if abs(c[0] - mn) < tol:
            c[0] = mn
        elif abs(c[0] - mx) < tol:
            c[0] = mx
        out.append(tuple(c))
        if (poly[i][0] - mn)*(nxt[0] - mn) < 0:
            t = (mn - poly[i][0]) / (nxt[0] - poly[i][0])
            y_ = poly[i][1] + t*(nxt[1] - poly[i][1])
            out.append((mn, y_))
        if (poly[i][0] - mx)*(nxt[0] - mx) < 0:
            t = (mx - poly[i][0]) / (nxt[0] - poly[i][0])
            y_ = poly[i][1] + t*(nxt[1] - poly[i][1])
            out.append((mx, y_))
    return out

class WingSegment:
    def __init__(self, root_chord, tip_chord, sweep_deg, span, x_start=0.0, y_start=0.0):
        self.root_chord = D(root_chord)
        self.tip_chord = D(tip_chord)
        self.sweep_rad = D(sweep_deg) * D(math.pi) / D("180")
        self.span = D(span)
        self.x_start = D(x_start)
        self.y_start = D(y_start)
        
        # В оригинальном коде: x_tip = x_start + span, y_tip = y_start + tan(sweep) * span
        self.x_tip = self.x_start + self.span
        self.y_tip = self.y_start + D(math.tan(float(self.sweep_rad))) * self.span

    def get_contour(self):
        # В оригинальном коде:
        # p0 = (x_start, y_start)
        # p1 = (x_start, y_start + root_chord)
        # p2 = (x_tip, y_tip + tip_chord)
        # p3 = (x_tip, y_tip)
        return [
            (self.x_start, self.y_start),
            (self.x_start, self.y_start + self.root_chord),
            (self.x_tip,   self.y_tip   + self.tip_chord),
            (self.x_tip,   self.y_tip)
        ]

def get_transformed_contour(segment, M, tol=None):
    if tol is None:
        tol = D("1e-6")
    pts = [apply_affine_transform(pt, M) for pt in segment.get_contour()]
    if len(pts) < 4:
        return pts
    P0, P1, P2, P3 = pts
    chord_diff = abs(D(segment.root_chord) - D(segment.tip_chord))
    if chord_diff > tol:
        # Выпрямляем по X (в оригинале было P1 = (P0[0], P1[1]) и P2 = (P3[0], P2[1]))
        P1 = (P0[0], P1[1])
        P2 = (P3[0], P2[1])
    return [P0, P1, P2, P3]

def transformed_diagonal_sac(poly, ext_factor=D("1.0"), tol=None, refine=True):
    if tol is None:
        tol = D("1e-6")
    if not poly:
        return None
    xs = [p[0] for p in poly]
    mn = min(xs)
    mx = max(xs)
    left_pts = [p for p in poly if abs(p[0] - mn) < tol]
    right_pts = [p for p in poly if abs(p[0] - mx) < tol]
    if not left_pts or not right_pts:
        return None
    lly = min(p[1] for p in left_pts)
    luy = max(p[1] for p in left_pts)
    rly = min(p[1] for p in right_pts)
    ruy = max(p[1] for p in right_pts)
    root_chord = luy - lly
    tip_chord = ruy - rly
    chord_diff = abs(root_chord - tip_chord)
    x_start, y_start = mn, lly
    x_tip, y_tip = mx, rly
    if chord_diff < tol:
        chord = (root_chord + tip_chord) / D("2")
        A_ext = (x_start, y_start - ext_factor * chord)
        B_ext = (x_start, y_start + chord + ext_factor * chord)
        C_ext = (x_tip, y_tip - ext_factor * chord)
        D_ext = (x_tip, y_tip + chord + ext_factor * chord)
    else:
        A_ext = (x_start, y_start - ext_factor * tip_chord)
        B_ext = (x_start, y_start + root_chord + ext_factor * tip_chord)
        C_ext = (x_tip, y_tip - ext_factor * root_chord)
        D_ext = (x_tip, y_tip + tip_chord + ext_factor * root_chord)
    m1, b1 = line_equation(A_ext, D_ext)
    m2, b2 = line_equation(C_ext, B_ext)
    sac = line_intersection(m1, b1, m2, b2)
    if not sac:
        if chord_diff < tol:
            center_x = (A_ext[0] + C_ext[0]) / D("2")
            center_y = (A_ext[1] + C_ext[1]) / D("2")
            sac = (center_x, center_y)
        else:
            return None
    if refine and sac:
        sac = refine_intersection(m1, b1, m2, b2, sac, iterations=10, tol=tol)
    return (A_ext, B_ext, C_ext, D_ext, sac)

def vertical_line_polygon_intersection(poly, x_val, tol=None):
    if tol is None:
        tol = D("1e-6")
    ys = []
    n = len(poly)
    for i in range(n):
        c, nx = poly[i], poly[(i+1)%n]
        x_c, x_n = c[0], nx[0]
        if (x_c - x_val) * (x_n - x_val) <= 0 and abs(x_n - x_c) > tol:
            t = (x_val - x_c) / (x_n - x_c)
            ys.append(c[1] + t * (nx[1] - c[1]))
    return sorted(list(set(ys)))

def vertical_sac_line_transformed(segment, M, pivot, boundary=D("0"), tol=None):
    if tol is None:
        tol = D("1e-6")
    poly = get_transformed_contour(segment, M)
    poly = clip_polygon_to_boundary(poly, boundary)
    poly = add_nose_extension_triangle_with_vertical_leg(poly)
    poly = verticalize_polygon(poly)
    res = transformed_diagonal_sac(poly)
    if not res:
        return [(D(0), D(0)), (D(0), D(0))]
    sac_pt = res[4]
    x_sac = sac_pt[0]
    ys = vertical_line_polygon_intersection(poly, x_sac)
    if len(ys) >= 2:
        return [(x_sac, ys[0]), (x_sac, ys[-1])]
    return [(x_sac, sac_pt[1]), (x_sac, sac_pt[1])]

def recursive_sac_merge(segments):
    if not segments:
        return []
    all_stages = [segments]
    curr = segments
    while len(curr) > 1:
        next_stage = []
        for i in range(0, len(curr) - 1, 2):
            s1, s2 = curr[i], curr[i+1]
            M_id = build_affine_matrix((D(0), D(0)), 0)
            sac1 = vertical_sac_line_transformed(s1, M_id, (D(0), D(0)))
            sac2 = vertical_sac_line_transformed(s2, M_id, (D(0), D(0)))
            
            x1, y1_lo, y1_hi = sac1[0][0], sac1[0][1], sac1[1][1]
            x2, y2_lo, y2_hi = sac2[0][0], sac2[0][1], sac2[1][1]
            
            L1, L2 = abs(y1_hi - y1_lo), abs(y2_hi - y2_lo)
            new_root = L1
            new_tip = L2
            new_span = abs(x2 - x1)
            
            # В WingSegment: x_tip = x_start + span, y_tip = y_start + tan(sweep) * span
            # Здесь x_start = x1, x_tip = x2 => span = x2 - x1
            # y_start = y1_lo, y_tip = y2_lo => dy = y2_lo - y1_lo = tan(sweep) * span
            if new_span != 0:
                sweep_rad = math.atan2(float(y2_lo - y1_lo), float(new_span))
                new_sweep_deg = math.degrees(sweep_rad)
            else:
                new_sweep_deg = 0
                
            new_seg = WingSegment(new_root, new_tip, new_sweep_deg, new_span, x1, y1_lo)
            next_stage.append(new_seg)
        if len(curr) % 2 == 1:
            next_stage.append(curr[-1])
        all_stages.append(next_stage)
        curr = next_stage
    return all_stages

def compute_total_area(segments):
    total = 0.0
    for seg in segments:
        rc, tc, sp = float(seg.root_chord), float(seg.tip_chord), float(seg.span)
        total += 0.5 * (rc + tc) * sp
    return total

def compute_mac_for_wing(segments):
    total_area = 0.0
    total_chord_area = 0.0
    for seg in segments:
        rc, tc, sp = float(seg.root_chord), float(seg.tip_chord), float(seg.span)
        seg_area = 0.5 * (rc + tc) * sp
        seg_chord = 0.5 * (rc + tc)
        total_area += seg_area
        total_chord_area += seg_area * seg_chord
    return total_chord_area / total_area if total_area > 1e-12 else 0.0

def compute_le_for_wing(segments):
    if not segments:
        return 0.0
    mn = float('inf')
    for seg in segments:
        for p in seg.get_contour():
            x_ = float(p[0])
            if x_ < mn: mn = x_
    return mn

def find_x25(segments):
    x_le = compute_le_for_wing(segments)
    mac_val = compute_mac_for_wing(segments)
    return x_le + 0.25 * mac_val
