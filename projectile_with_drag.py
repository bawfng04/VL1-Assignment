"""
=============================================================================
BÀI TẬP LỚN: XÁC ĐỊNH QUỸ ĐẠO CHUYỂN ĐỘNG NÉM XIÊN
TRONG TRỌNG TRƯỜNG CÓ LỰC CẢN MÔI TRƯỜNG
=============================================================================

1. PHƯƠNG TRÌNH CƠ BẢN
Định luật II Newton cho vật bay trong không khí là:
    m * a⃗ = m * g⃗ - h * v⃗

Lực tổng cộng làm vật chuyển động (m*a) chịu tác động bởi 2 lực:
- Lực kéo xuống của Trái Đất (m*g).
- Lực cản của gió đẩy ngược lại chiều bay (-h*v). Vật bay càng nhanh (v lớn) thì gió cản càng mạnh.

Trong đó:
    m  : khối lượng vật (kg)
    g  : gia tốc trọng trường (m/s²)
    h  : hệ số cản môi trường (kg/s)
    v⃗  : vận tốc của vật (m/s)
    a⃗  : gia tốc của vật (m/s²)

2. ĐIỀU KIỆN BAN ĐẦU:
Lúc vừa mới ném (t = 0):
- Vật nằm ở tay người ném: x = 0, y = 0
- Vận tốc bay theo chiều ngang: vx = v0 * cos(α)
- Vận tốc bay thẳng đứng lên: vy = v0 * sin(α)

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import (
    symbols, Function, Eq, dsolve, cos, sin, exp, simplify,
    lambdify, pi, Rational, latex, pprint, init_printing
)

# Khởi tạo chế độ hiển thị symbolic đẹp
init_printing(use_unicode=True)


# =============================================================================
# PHẦN 1: KHAI BÁO CÁC BIẾN VÀ THAM SỐ
# =============================================================================
print("=" * 70)
print("  CHƯƠNG TRÌNH TÍNH QUỸ ĐẠO NÉM XIÊN CÓ LỰC CẢN MÔI TRƯỜNG")
print("=" * 70)

# Khai báo biến symbolic
t = symbols('t', positive=True)
m_sym, g_sym, h_sym, v0_sym, alpha_sym = symbols(
    'm g h v_0 alpha', positive=True
)

# Khai báo hàm symbolic
x = Function('x')
y = Function('y')

# --- Nhập các thông số vật lý ---
# Giá trị mặc định (có thể thay đổi)
m_val = 1.0          # Khối lượng vật (kg)
g_val = 9.81         # Gia tốc trọng trường (m/s²)
h_val = 0.5          # Hệ số cản môi trường (kg/s)
v0_val = 50.0        # Vận tốc ban đầu (m/s)
t_flight = 15.0      # Thời gian bay tối đa (s)

# Các góc ném (độ)
angles_deg = [15, 30, 45, 60, 75]

print(f"\n--- Thông số đầu vào ---")
print(f"  Khối lượng vật      m  = {m_val} kg")
print(f"  Gia tốc trọng trường g  = {g_val} m/s²")
print(f"  Hệ số cản môi trường h  = {h_val} kg/s")
print(f"  Vận tốc ban đầu    v₀ = {v0_val} m/s")
print(f"  Thời gian bay tối đa t  = {t_flight} s")
print(f"  Các góc ném         α  = {angles_deg}°")


# =============================================================================
# PHẦN 2: THIẾT LẬP VÀ GIẢI HỆ PHƯƠNG TRÌNH VI PHÂN (SYMBOLIC)
# =============================================================================
print("\n" + "=" * 70)
print("  PHẦN 2: GIẢI HỆ PHƯƠNG TRÌNH VI PHÂN BẰNG SYMPY")
print("=" * 70)

# ---- CHUYỂN ĐỔI TOÁN HỌC VÀO CODE (PHƯƠNG NGANG Ox) ----
# Về mặt toán học:
# - Gia tốc a là đạo hàm bậc 2 của vị trí: x''(t)
# - Vận tốc v là đạo hàm bậc 1 của vị trí: x'(t)
# Theo phương ngang (Ox) không có trọng lực, phương trình rút gọn thành:
#   m * x''(t) = -h * x'(t)
# Chuyển vế sang: m * x''(t) + h * x'(t) = 0

print("\n--- Phương trình vi phân theo phương Ox ---")
# Trong Python, dùng hàm Eq (Equation) để viết phương trình trên.
# diff(t, 2) nghĩa là đạo hàm bậc 2. diff(t) là đạo hàm bậc 1.
ode_x = Eq(m_sym * x(t).diff(t, 2) + h_sym * x(t).diff(t), 0)
print(f"  m·x''(t) + h·x'(t) = 0")
pprint(ode_x)

# Gọi hàm dsolve() (Differential Solve) tìm ra công thức x(t)
# Mệnh đề ics (Initial Conditions) nạp điều kiện lúc mới ném.
sol_x = dsolve(
    ode_x,
    x(t),
    ics={
        x(0): 0,  # Vị trí ban đầu
        x(t).diff(t).subs(t, 0): v0_sym * cos(alpha_sym)  # Vận tốc ngang ban đầu
    }
)

print("\n  Nghiệm x(t):")
pprint(sol_x)

# ---- CHUYỂN ĐỔI TOÁN HỌC VÀO CODE (PHƯƠNG ĐỨNG Oy) ----
# Tương tự, theo phương đứng (Oy) có lực kéo của Trái Đất (-m*g):
#   m * y''(t) = -m*g - h * y'(t)
# Chuyển vế sang: m * y''(t) + h * y'(t) = -m*g

print("\n--- Phương trình vi phân theo phương Oy ---")
# Khai báo phương trình Oy vào máy tính
ode_y = Eq(m_sym * y(t).diff(t, 2) + h_sym * y(t).diff(t), -m_sym * g_sym)
print(f"  m·y''(t) + h·y'(t) = -m·g")
pprint(ode_y)

# Giải phương trình Oy
sol_y = dsolve(
    ode_y,
    y(t),
    ics={
        y(0): 0,  # Vị trí ban đầu
        y(t).diff(t).subs(t, 0): v0_sym * sin(alpha_sym)  # Vận tốc đứng ban đầu
    }
)

print("\n  Nghiệm y(t):")
pprint(sol_y)


# =============================================================================
# PHẦN 3: ĐƠN GIẢN HÓA NGHIỆM VÀ BIỂU DIỄN DẠNG TOÁN HỌC
# =============================================================================
print("\n" + "=" * 70)
print("  PHẦN 3: NGHIỆM GIẢI TÍCH SAU KHI ĐƠN GIẢN HÓA")
print("=" * 70)

# Lấy biểu thức vế phải của nghiệm
x_expr = simplify(sol_x.rhs)
y_expr = simplify(sol_y.rhs)

print("\n  x(t) = ")
pprint(x_expr)
print("\n  y(t) = ")
pprint(y_expr)

# Tính vận tốc theo từng phương
vx_expr = simplify(x_expr.diff(t))
vy_expr = simplify(y_expr.diff(t))

print("\n  vx(t) = dx/dt = ")
pprint(vx_expr)
print("\n  vy(t) = dy/dt = ")
pprint(vy_expr)


# =============================================================================
# PHẦN 4: TÍNH TOÁN SỐ V VÀ VẼ ĐỒ THỊ QUỸ ĐẠO
# =============================================================================
print("\n" + "=" * 70)
print("  PHẦN 4: TÍNH TOÁN SỐ VÀ VẼ ĐỒ THỊ")
print("=" * 70)

x_func = lambdify(
    (t, m_sym, g_sym, h_sym, v0_sym, alpha_sym), # Danh sách đầu vào
    x_expr,                                      # Công thức cần tính
    modules='numpy'                              # Dùng thư viện số học NumPy
)
y_func = lambdify(
    (t, m_sym, g_sym, h_sym, v0_sym, alpha_sym),
    y_expr,
    modules='numpy'
)
vx_func = lambdify(
    (t, m_sym, g_sym, h_sym, v0_sym, alpha_sym),
    vx_expr,
    modules='numpy'
)
vy_func = lambdify(
    (t, m_sym, g_sym, h_sym, v0_sym, alpha_sym),
    vy_expr,
    modules='numpy'
)

# Mảng thời gian
t_arr = np.linspace(0, t_flight, 2000)

# Định dạng màu sắc và nét vẽ cho từng góc
styles = {
    15: {'color': '#E74C3C', 'linestyle': '-',  'marker': '',  'label': r'$\alpha = 15°$'},
    30: {'color': '#3498DB', 'linestyle': '--', 'marker': '',  'label': r'$\alpha = 30°$'},
    45: {'color': '#2ECC71', 'linestyle': '-.',  'marker': '',  'label': r'$\alpha = 45°$'},
    60: {'color': '#F39C12', 'linestyle': ':',  'marker': '',  'label': r'$\alpha = 60°$'},
    75: {'color': '#9B59B6', 'linestyle': '-',  'marker': '',  'label': r'$\alpha = 75°$'},
}

# ---- Tạo Figure chính: Quỹ đạo y(x) ----
fig1, ax1 = plt.subplots(1, 1, figsize=(14, 8))

print("\n  Tính toán quỹ đạo cho các góc ném:")
print(f"  {'Góc (°)':>8} | {'Tầm xa (m)':>12} | {'Độ cao max (m)':>15} | {'Thời gian rơi (s)':>18}")
print(f"  {'-'*8}-+-{'-'*12}-+-{'-'*15}-+-{'-'*18}")

for alpha_deg in angles_deg:
    alpha_rad = np.deg2rad(alpha_deg)

    # Tính tọa độ x(t) và y(t)
    x_vals = x_func(t_arr, m_val, g_val, h_val, v0_val, alpha_rad)
    y_vals = y_func(t_arr, m_val, g_val, h_val, v0_val, alpha_rad)

    # Tìm thời điểm vật chạm đất (y = 0 lần thứ 2)
    # Chỉ lấy phần quỹ đạo có y >= 0 (bỏ qua y(0)=0)
    landing_indices = np.where((y_vals[1:] <= 0) & (y_vals[:-1] > 0))[0]
    if len(landing_indices) > 0:
        idx_land = landing_indices[0] + 1
        # Nội suy tuyến tính để tìm chính xác thời điểm chạm đất
        t_land = t_arr[idx_land - 1] + (0 - y_vals[idx_land - 1]) / \
                 (y_vals[idx_land] - y_vals[idx_land - 1]) * \
                 (t_arr[idx_land] - t_arr[idx_land - 1])
    else:
        idx_land = len(t_arr) - 1
        t_land = t_flight

    # Cắt dữ liệu đến khi chạm đất
    mask = t_arr <= t_land
    x_plot = x_vals[mask]
    y_plot = y_vals[mask]

    # Thêm điểm chạm đất chính xác
    x_land = x_func(t_land, m_val, g_val, h_val, v0_val, alpha_rad)
    y_land = 0.0
    x_plot = np.append(x_plot, x_land)
    y_plot = np.append(y_plot, y_land)

    # Tính các đại lượng đặc trưng
    range_val = x_land
    max_height = np.max(y_plot)

    print(f"  {alpha_deg:>8} | {range_val:>12.3f} | {max_height:>15.3f} | {t_land:>18.3f}")

    # Vẽ quỹ đạo
    style = styles[alpha_deg]
    ax1.plot(
        x_plot, y_plot,
        color=style['color'],
        linestyle=style['linestyle'],
        linewidth=2.5,
        label=style['label']
    )

    # Đánh dấu điểm rơi
    ax1.plot(x_land, 0, 'v', color=style['color'], markersize=10, zorder=5)

    # Đánh dấu điểm cao nhất
    idx_max = np.argmax(y_plot)
    ax1.plot(
        x_plot[idx_max], y_plot[idx_max], '*',
        color=style['color'], markersize=12, zorder=5
    )

# Định dạng đồ thị chính
ax1.set_xlabel('Khoảng cách ngang x (m)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Độ cao y (m)', fontsize=14, fontweight='bold')
ax1.set_title(
    'Quỹ đạo chuyển động ném xiên trong trọng trường có lực cản môi trường\n'
    f'(m = {m_val} kg, g = {g_val} m/s², h = {h_val} kg/s, v₀ = {v0_val} m/s)',
    fontsize=15, fontweight='bold', pad=15
)
ax1.legend(fontsize=13, loc='upper right', framealpha=0.9,
           edgecolor='gray', shadow=True)
ax1.set_xlim(left=0)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.axhline(y=0, color='black', linewidth=0.8)
ax1.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('quy_dao_nem_xien.png', dpi=300, bbox_inches='tight')
print("\n  [✓] Đã lưu đồ thị quỹ đạo: quy_dao_nem_xien.png")


# =============================================================================
# PHẦN 5: ĐỒ THỊ BỔ SUNG - SO SÁNH CÓ VÀ KHÔNG CÓ LỰC CẢN
# =============================================================================
print("\n" + "=" * 70)
print("  PHẦN 5: SO SÁNH QUỸ ĐẠO CÓ VÀ KHÔNG CÓ LỰC CẢN")
print("=" * 70)

fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))

# Quỹ đạo không có lực cản (h = 0): nghiệm giải tích cổ điển
# x(t) = v0 * cos(α) * t
# y(t) = v0 * sin(α) * t - 0.5 * g * t²
alpha_compare = 45  # Góc so sánh (độ)
alpha_compare_rad = np.deg2rad(alpha_compare)

# --- Có lực cản ---
x_drag = x_func(t_arr, m_val, g_val, h_val, v0_val, alpha_compare_rad)
y_drag = y_func(t_arr, m_val, g_val, h_val, v0_val, alpha_compare_rad)

# Cắt đến khi chạm đất
landing_drag = np.where((y_drag[1:] <= 0) & (y_drag[:-1] > 0))[0]
if len(landing_drag) > 0:
    idx_d = landing_drag[0] + 1
    t_land_drag = t_arr[idx_d - 1] + (0 - y_drag[idx_d - 1]) / \
                  (y_drag[idx_d] - y_drag[idx_d - 1]) * \
                  (t_arr[idx_d] - t_arr[idx_d - 1])
else:
    t_land_drag = t_flight

mask_drag = t_arr <= t_land_drag
x_drag_plot = x_drag[mask_drag]
y_drag_plot = y_drag[mask_drag]

# --- Không có lực cản ---
x_no_drag = v0_val * np.cos(alpha_compare_rad) * t_arr
y_no_drag = v0_val * np.sin(alpha_compare_rad) * t_arr - 0.5 * g_val * t_arr**2

landing_no_drag = np.where((y_no_drag[1:] <= 0) & (y_no_drag[:-1] > 0))[0]
if len(landing_no_drag) > 0:
    idx_nd = landing_no_drag[0] + 1
    t_land_no = t_arr[idx_nd - 1] + (0 - y_no_drag[idx_nd - 1]) / \
                (y_no_drag[idx_nd] - y_no_drag[idx_nd - 1]) * \
                (t_arr[idx_nd] - t_arr[idx_nd - 1])
else:
    t_land_no = t_flight

mask_no = t_arr <= t_land_no
x_no_plot = x_no_drag[mask_no]
y_no_plot = y_no_drag[mask_no]

# Vẽ so sánh
ax2.plot(x_no_plot, y_no_plot, 'b-', linewidth=2.5,
         label=f'Không có lực cản (h = 0)')
ax2.plot(x_drag_plot, y_drag_plot, 'r--', linewidth=2.5,
         label=f'Có lực cản (h = {h_val} kg/s)')

# Tô vùng chênh lệch
from matplotlib.patches import FancyArrowPatch
ax2.annotate(
    '', xy=(x_no_plot[-1], 0), xytext=(x_drag_plot[-1], 0),
    arrowprops=dict(arrowstyle='<->', color='green', lw=2)
)
range_diff = x_no_plot[-1] - x_drag_plot[-1]
mid_x = (x_no_plot[-1] + x_drag_plot[-1]) / 2
ax2.text(mid_x, -8, f'ΔR = {range_diff:.1f} m',
         ha='center', fontsize=12, color='green', fontweight='bold')

ax2.set_xlabel('Khoảng cách ngang x (m)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Độ cao y (m)', fontsize=14, fontweight='bold')
ax2.set_title(
    f'So sánh quỹ đạo có và không có lực cản (α = {alpha_compare}°)\n'
    f'(m = {m_val} kg, g = {g_val} m/s², v₀ = {v0_val} m/s)',
    fontsize=15, fontweight='bold', pad=15
)
ax2.legend(fontsize=13, loc='upper right', framealpha=0.9,
           edgecolor='gray', shadow=True)
ax2.set_xlim(left=0)
ax2.set_ylim(bottom=-15)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('so_sanh_luc_can.png', dpi=300, bbox_inches='tight')
print("  [✓] Đã lưu đồ thị so sánh: so_sanh_luc_can.png")


# =============================================================================
# PHẦN 6: ĐỒ THỊ VẬN TỐC THEO THỜI GIAN
# =============================================================================
print("\n" + "=" * 70)
print("  PHẦN 6: ĐỒ THỊ VẬN TỐC THEO THỜI GIAN")
print("=" * 70)

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 7))

for alpha_deg in angles_deg:
    alpha_rad = np.deg2rad(alpha_deg)

    vx_vals = vx_func(t_arr, m_val, g_val, h_val, v0_val, alpha_rad)
    vy_vals = vy_func(t_arr, m_val, g_val, h_val, v0_val, alpha_rad)
    y_vals = y_func(t_arr, m_val, g_val, h_val, v0_val, alpha_rad)

    # Tìm thời gian chạm đất
    landing = np.where((y_vals[1:] <= 0) & (y_vals[:-1] > 0))[0]
    if len(landing) > 0:
        idx_l = landing[0] + 1
        t_land_v = t_arr[idx_l - 1] + (0 - y_vals[idx_l - 1]) / \
                   (y_vals[idx_l] - y_vals[idx_l - 1]) * \
                   (t_arr[idx_l] - t_arr[idx_l - 1])
    else:
        t_land_v = t_flight

    mask_v = t_arr <= t_land_v
    t_plot = t_arr[mask_v]
    vx_plot = vx_vals[mask_v]
    vy_plot = vy_vals[mask_v]

    style = styles[alpha_deg]

    ax3a.plot(t_plot, vx_plot, color=style['color'],
              linestyle=style['linestyle'], linewidth=2, label=style['label'])
    ax3b.plot(t_plot, vy_plot, color=style['color'],
              linestyle=style['linestyle'], linewidth=2, label=style['label'])

ax3a.set_xlabel('Thời gian t (s)', fontsize=13, fontweight='bold')
ax3a.set_ylabel('vₓ (m/s)', fontsize=13, fontweight='bold')
ax3a.set_title('Vận tốc theo phương ngang vₓ(t)', fontsize=14, fontweight='bold')
ax3a.legend(fontsize=11, framealpha=0.9)
ax3a.grid(True, alpha=0.3, linestyle='--')
ax3a.tick_params(axis='both', labelsize=11)

ax3b.set_xlabel('Thời gian t (s)', fontsize=13, fontweight='bold')
ax3b.set_ylabel('vᵧ (m/s)', fontsize=13, fontweight='bold')
ax3b.set_title('Vận tốc theo phương đứng vᵧ(t)', fontsize=14, fontweight='bold')
ax3b.legend(fontsize=11, framealpha=0.9)
ax3b.grid(True, alpha=0.3, linestyle='--')
ax3b.tick_params(axis='both', labelsize=11)

plt.tight_layout()
plt.savefig('van_toc_theo_thoi_gian.png', dpi=300, bbox_inches='tight')
print("  [✓] Đã lưu đồ thị vận tốc: van_toc_theo_thoi_gian.png")


# =============================================================================
# PHẦN 7: ĐỒ THỊ NĂNG LƯỢNG THEO THỜI GIAN
# =============================================================================
print("\n" + "=" * 70)
print("  PHẦN 7: ĐỒ THỊ NĂNG LƯỢNG THEO THỜI GIAN (α = 45°)")
print("=" * 70)

fig4, ax4 = plt.subplots(1, 1, figsize=(14, 8))

alpha_e = np.deg2rad(45)
x_e = x_func(t_arr, m_val, g_val, h_val, v0_val, alpha_e)
y_e = y_func(t_arr, m_val, g_val, h_val, v0_val, alpha_e)
vx_e = vx_func(t_arr, m_val, g_val, h_val, v0_val, alpha_e)
vy_e = vy_func(t_arr, m_val, g_val, h_val, v0_val, alpha_e)

# Tìm thời gian chạm đất
landing_e = np.where((y_e[1:] <= 0) & (y_e[:-1] > 0))[0]
if len(landing_e) > 0:
    idx_e = landing_e[0] + 1
    t_land_e = t_arr[idx_e - 1] + (0 - y_e[idx_e - 1]) / \
               (y_e[idx_e] - y_e[idx_e - 1]) * \
               (t_arr[idx_e] - t_arr[idx_e - 1])
else:
    t_land_e = t_flight

mask_e = t_arr <= t_land_e
t_e_plot = t_arr[mask_e]
y_e_plot = y_e[mask_e]
vx_e_plot = vx_e[mask_e]
vy_e_plot = vy_e[mask_e]

# Tính các loại năng lượng
KE = 0.5 * m_val * (vx_e_plot**2 + vy_e_plot**2)  # Động năng
PE = m_val * g_val * y_e_plot                       # Thế năng
ME = KE + PE                                         # Cơ năng
E0 = 0.5 * m_val * v0_val**2                        # Năng lượng ban đầu
E_lost = E0 - ME                                     # Năng lượng mất do lực cản

ax4.plot(t_e_plot, KE, 'r-', linewidth=2.5, label='Động năng (KE)')
ax4.plot(t_e_plot, PE, 'b--', linewidth=2.5, label='Thế năng (PE)')
ax4.plot(t_e_plot, ME, 'g-.', linewidth=2.5, label='Cơ năng (ME = KE + PE)')
ax4.plot(t_e_plot, E_lost, 'm:', linewidth=2.5, label='Năng lượng mất do lực cản')
ax4.axhline(y=E0, color='gray', linewidth=1, linestyle='--', alpha=0.7,
            label=f'Năng lượng ban đầu E₀ = {E0:.1f} J')

ax4.set_xlabel('Thời gian t (s)', fontsize=14, fontweight='bold')
ax4.set_ylabel('Năng lượng (J)', fontsize=14, fontweight='bold')
ax4.set_title(
    f'Biến thiên năng lượng theo thời gian (α = 45°)\n'
    f'(m = {m_val} kg, g = {g_val} m/s², h = {h_val} kg/s, v₀ = {v0_val} m/s)',
    fontsize=15, fontweight='bold', pad=15
)
ax4.legend(fontsize=12, loc='center right', framealpha=0.9,
           edgecolor='gray', shadow=True)
ax4.grid(True, alpha=0.3, linestyle='--')
ax4.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('nang_luong_theo_thoi_gian.png', dpi=300, bbox_inches='tight')
print("  [✓] Đã lưu đồ thị năng lượng: nang_luong_theo_thoi_gian.png")


# =============================================================================
# PHẦN 8: ĐỒ THỊ ẢNH HƯỞNG CỦA HỆ SỐ CẢN h
# =============================================================================
print("\n" + "=" * 70)
print("  PHẦN 8: ẢNH HƯỞNG CỦA HỆ SỐ CẢN h ĐẾN QUỸ ĐẠO (α = 45°)")
print("=" * 70)

fig5, ax5 = plt.subplots(1, 1, figsize=(14, 8))

h_values = [0, 0.1, 0.3, 0.5, 1.0, 2.0]
h_colors = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']
h_styles = ['-', '--', '-.', ':', '-', '--']

alpha_h = np.deg2rad(45)

for i, h_v in enumerate(h_values):
    if h_v == 0:
        # Nghiệm giải tích cho h = 0
        x_h = v0_val * np.cos(alpha_h) * t_arr
        y_h = v0_val * np.sin(alpha_h) * t_arr - 0.5 * g_val * t_arr**2
    else:
        x_h = x_func(t_arr, m_val, g_val, h_v, v0_val, alpha_h)
        y_h = y_func(t_arr, m_val, g_val, h_v, v0_val, alpha_h)

    # Cắt đến khi chạm đất
    landing_h = np.where((y_h[1:] <= 0) & (y_h[:-1] > 0))[0]
    if len(landing_h) > 0:
        idx_h = landing_h[0] + 1
        t_land_h = t_arr[idx_h - 1] + (0 - y_h[idx_h - 1]) / \
                   (y_h[idx_h] - y_h[idx_h - 1]) * \
                   (t_arr[idx_h] - t_arr[idx_h - 1])
    else:
        t_land_h = t_flight

    mask_h = t_arr <= t_land_h
    x_h_plot = x_h[mask_h]
    y_h_plot = y_h[mask_h]

    label_h = f'h = {h_v} kg/s' if h_v > 0 else 'h = 0 (không cản)'
    ax5.plot(x_h_plot, y_h_plot, color=h_colors[i],
             linestyle=h_styles[i], linewidth=2.5, label=label_h)

ax5.set_xlabel('Khoảng cách ngang x (m)', fontsize=14, fontweight='bold')
ax5.set_ylabel('Độ cao y (m)', fontsize=14, fontweight='bold')
ax5.set_title(
    f'Ảnh hưởng của hệ số cản h đến quỹ đạo (α = 45°)\n'
    f'(m = {m_val} kg, g = {g_val} m/s², v₀ = {v0_val} m/s)',
    fontsize=15, fontweight='bold', pad=15
)
ax5.legend(fontsize=12, loc='upper right', framealpha=0.9,
           edgecolor='gray', shadow=True)
ax5.set_xlim(left=0)
ax5.set_ylim(bottom=0)
ax5.grid(True, alpha=0.3, linestyle='--')
ax5.axhline(y=0, color='black', linewidth=0.8)
ax5.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig('anh_huong_he_so_can.png', dpi=300, bbox_inches='tight')
print("  [✓] Đã lưu đồ thị ảnh hưởng hệ số cản: anh_huong_he_so_can.png")


# =============================================================================
# PHẦN 9: IN NGHIỆM GIẢI TÍCH DẠNG LATEX
# =============================================================================
print("\n" + "=" * 70)
print("  PHẦN 9: NGHIỆM GIẢI TÍCH DẠNG LATEX")
print("=" * 70)

print(f"\n  x(t) = {latex(x_expr)}")
print(f"\n  y(t) = {latex(y_expr)}")
print(f"\n  vx(t) = {latex(vx_expr)}")
print(f"\n  vy(t) = {latex(vy_expr)}")

# Hiển thị tất cả đồ thị
plt.show()

print("\n" + "=" * 70)
print("  CHƯƠNG TRÌNH HOÀN TẤT!")
print("=" * 70)
