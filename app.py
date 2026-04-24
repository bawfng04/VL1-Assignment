"""
Ứng dụng tương tác: Quỹ đạo ném xiên trong trọng trường có lực cản môi trường

1. PHƯƠNG TRÌNH CƠ BẢN
Định luật II Newton cho vật bay trong không khí là:
    m * a = m * g - h * v

Lực tổng cộng làm vật chuyển động (m*a) chịu tác động bởi 2 lực:
- Lực kéo xuống của Trái Đất (m*g).
- Lực cản của gió đẩy ngược lại chiều bay (-h*v). Vật bay càng nhanh (v lớn) thì gió cản càng mạnh.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sympy import (
    symbols, Function, Eq, dsolve, cos, sin, simplify,
    lambdify, latex
)


st.set_page_config(
    page_title="Quỹ Đạo Ném Xiên | BTL Vật Lý 1",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# phần này tạo UI nên khỏi đọc cũng được
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp {
        font-family: 'Inter', sans-serif;
        color: #1e293b;
    }

    .page-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
        border-bottom: 2px solid #2563eb;
        padding-bottom: 0.5rem;
        margin-bottom: 0.3rem;
    }

    .param-bar {
        display: flex; gap: 0.6rem; flex-wrap: wrap;
        margin-bottom: 1rem;
    }
    .param-chip {
        background: #f1f5f9;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        padding: 0.3rem 0.7rem;
        font-size: 0.82rem;
        color: #334155;
    }
    .param-chip b { color: #1e40af; }

    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #0f172a;
        margin: 1rem 0 0.6rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 1px solid #e2e8f0;
    }

    .data-table {
        width: 100%;
        border-collapse: collapse;
        margin: 0.8rem 0;
        font-size: 0.85rem;
    }
    .data-table th {
        background: #1e40af;
        color: #ffffff;
        padding: 8px 12px;
        font-weight: 600;
        text-align: left;
    }
    .data-table td {
        padding: 7px 12px;
        border-bottom: 1px solid #e2e8f0;
        color: #1e293b;
    }
    .data-table tr:nth-child(even) td { background: #f8fafc; }
    .data-table tr:hover td { background: #eff6ff; }

    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# giải phương trình vi phân bằng SymPy
# ============================================================
@st.cache_resource
def solve_ode_symbolic():
    t_s = symbols('t', positive=True)
    m_s, g_s, h_s, v0_s, a_s = symbols('m g h v_0 alpha', positive=True)
    xf, yf = Function('x'), Function('y')

    # ---- CHUYỂN ĐỔI TOÁN HỌC VÀO CODE ----
    # Gia tốc a là đạo hàm bậc 2 của vị trí: x''(t) và y''(t)
    # Vận tốc v là đạo hàm bậc 1 của vị trí: x'(t) và y'(t)
    
    # 1. Giải phương trình theo phương ngang (Ox)
    # m*x''(t) + h*x'(t) = 0
    sol_x = dsolve(
        Eq(m_s * xf(t_s).diff(t_s, 2) + h_s * xf(t_s).diff(t_s), 0),
        xf(t_s),
        ics={xf(0): 0, xf(t_s).diff(t_s).subs(t_s, 0): v0_s * cos(a_s)} # Điều kiện: x(0)=0, vx(0)=v0*cos(a)
    )
    
    # 2. Giải phương trình theo phương đứng (Oy)
    # m*y''(t) + h*y'(t) = -m*g
    sol_y = dsolve(
        Eq(m_s * yf(t_s).diff(t_s, 2) + h_s * yf(t_s).diff(t_s), -m_s * g_s),
        yf(t_s),
        ics={yf(0): 0, yf(t_s).diff(t_s).subs(t_s, 0): v0_s * sin(a_s)} # Điều kiện: y(0)=0, vy(0)=v0*sin(a)
    )

    x_expr = simplify(sol_x.rhs)
    y_expr = simplify(sol_y.rhs)
    vx_expr = simplify(x_expr.diff(t_s))
    vy_expr = simplify(y_expr.diff(t_s))

    args = (t_s, m_s, g_s, h_s, v0_s, a_s)
    
    return (
        lambdify(args, x_expr, 'numpy'),
        lambdify(args, y_expr, 'numpy'),
        lambdify(args, vx_expr, 'numpy'),
        lambdify(args, vy_expr, 'numpy'),
        latex(x_expr), latex(y_expr), latex(vx_expr), latex(vy_expr),
    )


# ============================================================
# HÀM TÍNH QUỸ ĐẠO
# ============================================================
def compute_trajectory(x_fn, y_fn, vx_fn, vy_fn, m, g, h, v0, alpha_deg, t_max):
    a = np.deg2rad(alpha_deg)
    t = np.linspace(0, t_max, 3000)
    x = x_fn(t, m, g, h, v0, a)
    y = y_fn(t, m, g, h, v0, a)
    vx = vx_fn(t, m, g, h, v0, a)
    vy = vy_fn(t, m, g, h, v0, a)

    land = np.where((y[1:] <= 0) & (y[:-1] > 0))[0]
    if len(land) > 0:
        i = land[0] + 1
        tl = t[i-1] + (0 - y[i-1]) / (y[i] - y[i-1]) * (t[i] - t[i-1])
    else:
        tl = t_max

    mask = t <= tl
    tc, xc, yc = t[mask], x[mask], y[mask]
    vxc, vyc = vx[mask], vy[mask]

    xl = x_fn(tl, m, g, h, v0, a)
    vxl = vx_fn(tl, m, g, h, v0, a)
    vyl = vy_fn(tl, m, g, h, v0, a)

    tc = np.append(tc, tl)
    xc = np.append(xc, xl)
    yc = np.append(yc, 0.0)
    vxc = np.append(vxc, vxl)
    vyc = np.append(vyc, vyl)

    return dict(t=tc, x=xc, y=yc, vx=vxc, vy=vyc,
                t_land=tl, range=xl, max_height=np.max(yc),
                x_at_max=xc[np.argmax(yc)], t_at_max=tc[np.argmax(yc)],
                v_impact=np.sqrt(vxl**2 + vyl**2))


def compute_no_drag(v0, alpha_deg, g, t_max):
    a = np.deg2rad(alpha_deg)
    t = np.linspace(0, t_max, 3000)
    x = v0 * np.cos(a) * t
    y = v0 * np.sin(a) * t - 0.5 * g * t**2
    land = np.where((y[1:] <= 0) & (y[:-1] > 0))[0]
    tl = (t[land[0]] + (0 - y[land[0]]) / (y[land[0]+1] - y[land[0]]) *
          (t[land[0]+1] - t[land[0]])) if len(land) > 0 else t_max
    mask = t <= tl
    xl = v0 * np.cos(a) * tl
    return dict(t=np.append(t[mask], tl), x=np.append(x[mask], xl),
                y=np.append(y[mask], 0.0), range=xl,
                max_height=(v0*np.sin(a))**2/(2*g), t_land=tl)


# phần này tạo plot, bỏ qua cũng đc
LAYOUT = dict(
    template="plotly_white",
    font=dict(family="Inter, sans-serif", size=13, color="#1e293b"),
    legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.95)",
                bordercolor="#e2e8f0", borderwidth=1),
    margin=dict(l=55, r=25, t=50, b=55),
    hoverlabel=dict(font_size=12, font_family="Inter"),
    paper_bgcolor="white",
    plot_bgcolor="white",
)
COLORS = ['#2563eb', '#dc2626', '#16a34a', '#ea580c', '#7c3aed',
          '#0891b2', '#c026d3', '#4f46e5']
DASHES = ['solid', 'dash', 'dashdot', 'dot', 'longdash',
          'longdashdot', 'solid', 'dash']



# Giải phương trình vi phân bằng SymPy
x_fn, y_fn, vx_fn, vy_fn, x_ltx, y_ltx, vx_ltx, vy_ltx = solve_ode_symbolic()


# UI nhập thông số
with st.sidebar:
    st.markdown("### Thông số vật lý")
    m  = st.slider("Khối lượng m (kg)",              0.1, 10.0, 1.0, 0.1)
    g  = st.slider("Gia tốc trọng trường g (m/s²)",  1.0, 20.0, 9.81, 0.01)
    h  = st.slider("Hệ số cản h (kg/s)",             0.0,  5.0, 0.5, 0.05)
    v0 = st.slider("Vận tốc ban đầu v₀ (m/s)",      5.0, 200.0, 50.0, 1.0)
    t_max = st.slider("Thời gian tối đa t (s)",      1.0, 60.0, 15.0, 0.5)

    st.markdown("---")
    st.markdown("### Góc ném α")
    angle_mode = st.radio(
        "Chế độ chọn góc:",
        ["Góc cố định (đề bài)", "Thanh trượt tùy chỉnh", "Nhập tự do"]
    )

    if angle_mode == "Góc cố định (đề bài)":
        selected = st.multiselect(
            "Chọn các góc (°):",
            options=list(range(5, 86, 5)),
            default=[15, 30, 45, 60, 75]
        )
    elif angle_mode == "Thanh trượt tùy chỉnh":
        sa = st.slider("Góc ném α (°)", 1, 89, 45, 1)
        show_ref = st.checkbox("Hiển thị thêm 15°, 45°, 75° tham chiếu", value=True)
        selected = [sa] + ([a for a in [15, 45, 75] if a != sa] if show_ref else [])
    else:
        txt = st.text_input(
            "Nhập các góc (phân cách bằng dấu phẩy):",
            "15, 30, 45, 60, 75"
        )
        try:
            selected = sorted(set(
                int(a.strip()) for a in txt.split(",")
                if a.strip() and 1 <= int(a.strip()) <= 89
            ))
        except ValueError:
            selected = [45]
            st.error("Vui lòng nhập số nguyên hợp lệ từ 1 đến 89.")

    st.markdown("---")
    st.caption("Bài tập lớn Vật Lý 1 — Python + SymPy + Streamlit")


st.markdown(
    '<div class="page-title">'
    'Quỹ đạo chuyển động ném xiên trong trọng trường có lực cản môi trường'
    '</div>',
    unsafe_allow_html=True
)
st.latex(r"m\vec{a} = m\vec{g} - h\vec{v}")

chips = (
    f'<span class="param-chip"><b>m</b> = {m} kg</span>'
    f'<span class="param-chip"><b>g</b> = {g} m/s²</span>'
    f'<span class="param-chip"><b>h</b> = {h} kg/s</span>'
    f'<span class="param-chip"><b>v₀</b> = {v0} m/s</span>'
    f'<span class="param-chip"><b>t</b> = {t_max} s</span>'
    f'<span class="param-chip"><b>α</b> = '
    f'{", ".join(str(a) + "°" for a in selected)}</span>'
)
st.markdown(f'<div class="param-bar">{chips}</div>', unsafe_allow_html=True)


# ============================================================
# TÍNH TOÁN QUỸ ĐẠO
# ============================================================
trajs = {
    a: compute_trajectory(x_fn, y_fn, vx_fn, vy_fn, m, g, h, v0, a, t_max)
    for a in selected
}

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Quỹ đạo", "So sánh lực cản", "Vận tốc",
    "Năng lượng", "Khảo sát hệ số h", "Lý thuyết"
])

with tab1:
    st.markdown(
        '<div class="section-title">Quỹ đạo chất điểm y(x)</div>',
        unsafe_allow_html=True
    )
    fig1 = go.Figure()
    for i, a in enumerate(selected):
        tr = trajs[a]
        c, d = COLORS[i % len(COLORS)], DASHES[i % len(DASHES)]
        fig1.add_trace(go.Scatter(
            x=tr['x'], y=tr['y'], mode='lines',
            name=f'α = {a}°',
            line=dict(color=c, width=2.5, dash=d),
            hovertemplate=(
                f'<b>α = {a}°</b><br>'
                'x = %{x:.1f} m<br>'
                'y = %{y:.1f} m<extra></extra>'
            )
        ))
        im = np.argmax(tr['y'])
        fig1.add_trace(go.Scatter(
            x=[tr['x'][im]], y=[tr['y'][im]], mode='markers',
            marker=dict(symbol='diamond', size=9, color=c,
                        line=dict(width=1, color='white')),
            showlegend=False,
            hovertemplate=(
                f'<b>Đỉnh α = {a}°</b><br>'
                f'H = {tr["max_height"]:.1f} m<extra></extra>'
            )
        ))
        fig1.add_trace(go.Scatter(
            x=[tr['range']], y=[0], mode='markers',
            marker=dict(symbol='triangle-down', size=9, color=c,
                        line=dict(width=1, color='white')),
            showlegend=False,
            hovertemplate=(
                f'<b>Chạm đất α = {a}°</b><br>'
                f'R = {tr["range"]:.1f} m<extra></extra>'
            )
        ))

    fig1.update_layout(
        **LAYOUT, height=500,
        title=dict(text="Quỹ đạo ném xiên có lực cản", font=dict(size=15)),
        xaxis_title="x (m)", yaxis_title="y (m)",
        xaxis=dict(rangemode='tozero', gridcolor='#f1f5f9'),
        yaxis=dict(rangemode='tozero', gridcolor='#f1f5f9'),
    )
    st.plotly_chart(fig1, width="stretch")

    # Bảng số liệu
    st.markdown(
        '<div class="section-title">Bảng số liệu chi tiết</div>',
        unsafe_allow_html=True
    )
    rows = ""
    for a in selected:
        tr = trajs[a]
        v0x = v0 * np.cos(np.deg2rad(a))
        v0y = v0 * np.sin(np.deg2rad(a))
        rows += (
            f"<tr><td><b>{a}°</b></td>"
            f"<td>{tr['range']:.2f}</td>"
            f"<td>{tr['max_height']:.2f}</td>"
            f"<td>{tr['t_land']:.3f}</td>"
            f"<td>{v0x:.2f}</td>"
            f"<td>{v0y:.2f}</td>"
            f"<td>{tr['v_impact']:.2f}</td></tr>"
        )
    st.markdown(
        f"""<table class="data-table"><thead><tr>
        <th>Góc α</th><th>Tầm xa R (m)</th><th>Độ cao max H (m)</th>
        <th>Thời gian bay (s)</th><th>v₀ₓ (m/s)</th><th>v₀ᵧ (m/s)</th>
        <th>v chạm đất (m/s)</th>
        </tr></thead><tbody>{rows}</tbody></table>""",
        unsafe_allow_html=True
    )


# ================================================================
# TAB 2: SO SÁNH LỰC CẢN
# ================================================================
with tab2:
    st.markdown(
        '<div class="section-title">So sánh quỹ đạo có và không có lực cản</div>',
        unsafe_allow_html=True
    )

    compare_angle = st.select_slider(
        "Chọn góc so sánh (°):",
        options=list(range(5, 86)),
        value=selected[0] if selected else 45,
        key="cmp_angle"
    )

    td = compute_trajectory(x_fn, y_fn, vx_fn, vy_fn, m, g, h, v0, compare_angle, t_max)
    tn = compute_no_drag(v0, compare_angle, g, t_max)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=tn['x'], y=tn['y'], mode='lines',
        name='Không cản (h = 0)',
        line=dict(color='#2563eb', width=2.5),
        fill='tozeroy', fillcolor='rgba(37,99,235,0.06)'
    ))
    fig2.add_trace(go.Scatter(
        x=td['x'], y=td['y'], mode='lines',
        name=f'Có cản (h = {h})',
        line=dict(color='#dc2626', width=2.5, dash='dash'),
        fill='tozeroy', fillcolor='rgba(220,38,38,0.06)'
    ))

    dR = tn['range'] - td['range']
    dH = tn['max_height'] - td['max_height']

    fig2.update_layout(
        **LAYOUT, height=480,
        title=dict(
            text=f"α = {compare_angle}°  |  ΔR = {dR:.1f} m,  ΔH = {dH:.1f} m",
            font=dict(size=14)
        ),
        xaxis_title="x (m)", yaxis_title="y (m)",
        xaxis=dict(rangemode='tozero', gridcolor='#f1f5f9'),
        yaxis=dict(rangemode='tozero', gridcolor='#f1f5f9'),
    )
    st.plotly_chart(fig2, width="stretch")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R (không cản)", f"{tn['range']:.1f} m")
    c2.metric("R (có cản)", f"{td['range']:.1f} m",
              delta=f"-{dR:.1f} m", delta_color="inverse")
    c3.metric("H (không cản)", f"{tn['max_height']:.1f} m")
    c4.metric("H (có cản)", f"{td['max_height']:.1f} m",
              delta=f"-{dH:.1f} m", delta_color="inverse")


# ================================================================
# TAB 3: VẬN TỐC
# ================================================================
with tab3:
    st.markdown(
        '<div class="section-title">Biến thiên vận tốc theo thời gian</div>',
        unsafe_allow_html=True
    )

    fig3 = make_subplots(
        rows=1, cols=3,
        subplot_titles=("Vận tốc ngang vₓ(t)", "Vận tốc đứng vᵧ(t)", "Tốc độ |v|(t)"),
        horizontal_spacing=0.07
    )

    for i, a in enumerate(selected):
        tr = trajs[a]
        c, d = COLORS[i % len(COLORS)], DASHES[i % len(DASHES)]
        sp = np.sqrt(tr['vx']**2 + tr['vy']**2)
        fig3.add_trace(go.Scatter(
            x=tr['t'], y=tr['vx'], mode='lines',
            name=f'α = {a}°', line=dict(color=c, width=2, dash=d),
            legendgroup=f'g{a}'), row=1, col=1)
        fig3.add_trace(go.Scatter(
            x=tr['t'], y=tr['vy'], mode='lines',
            line=dict(color=c, width=2, dash=d), legendgroup=f'g{a}',
            showlegend=False), row=1, col=2)
        fig3.add_trace(go.Scatter(
            x=tr['t'], y=sp, mode='lines',
            line=dict(color=c, width=2, dash=d), legendgroup=f'g{a}',
            showlegend=False), row=1, col=3)

    fig3.update_xaxes(title_text="t (s)", gridcolor='#f1f5f9')
    fig3.update_yaxes(title_text="m/s", row=1, col=1, gridcolor='#f1f5f9')
    fig3.update_yaxes(gridcolor='#f1f5f9')
    fig3.update_layout(
        **LAYOUT, height=440,
        title=dict(text="Biến thiên vận tốc theo thời gian", font=dict(size=15))
    )
    st.plotly_chart(fig3, width="stretch")


# ================================================================
# TAB 4: NĂNG LƯỢNG
# ================================================================
with tab4:
    st.markdown(
        '<div class="section-title">Biến thiên năng lượng theo thời gian</div>',
        unsafe_allow_html=True
    )

    energy_angle = st.select_slider(
        "Chọn góc phân tích (°):",
        options=list(range(5, 86)),
        value=selected[0] if selected else 45,
        key="energy_angle"
    )

    te = compute_trajectory(x_fn, y_fn, vx_fn, vy_fn, m, g, h, v0, energy_angle, t_max)
    KE = 0.5 * m * (te['vx']**2 + te['vy']**2)
    PE = m * g * te['y']
    ME = KE + PE
    E0 = 0.5 * m * v0**2
    EL = E0 - ME

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=te['t'], y=KE, mode='lines', name='Động năng (KE)',
        line=dict(color='#dc2626', width=2.5)))
    fig4.add_trace(go.Scatter(
        x=te['t'], y=PE, mode='lines', name='Thế năng (PE)',
        line=dict(color='#2563eb', width=2.5, dash='dash')))
    fig4.add_trace(go.Scatter(
        x=te['t'], y=ME, mode='lines', name='Cơ năng (KE+PE)',
        line=dict(color='#16a34a', width=2.5, dash='dashdot')))
    fig4.add_trace(go.Scatter(
        x=te['t'], y=EL, mode='lines', name='Mất do lực cản',
        line=dict(color='#7c3aed', width=2.5, dash='dot')))
    fig4.add_hline(
        y=E0, line_dash="dot", line_color="#94a3b8",
        annotation_text=f"E₀ = {E0:.1f} J"
    )

    fig4.update_layout(
        **LAYOUT, height=480,
        title=dict(
            text=f"Biến thiên năng lượng (α = {energy_angle}°)",
            font=dict(size=15)
        ),
        xaxis_title="t (s)", yaxis_title="Năng lượng (J)",
        xaxis=dict(gridcolor='#f1f5f9'),
        yaxis=dict(gridcolor='#f1f5f9'),
    )
    st.plotly_chart(fig4, width="stretch")

    p1, p2, p3 = st.columns(3)
    p1.metric("Năng lượng ban đầu E₀", f"{E0:.1f} J")
    p2.metric("Động năng khi chạm đất", f"{KE[-1]:.1f} J")
    pct = f" ({EL[-1]/E0*100:.0f}%)" if E0 > 0 else ""
    p3.metric("Mất do lực cản", f"{EL[-1]:.1f} J{pct}")


# ================================================================
# TAB 5: KHẢO SÁT HỆ SỐ h
# ================================================================
with tab5:
    st.markdown(
        '<div class="section-title">Ảnh hưởng của hệ số cản h đến quỹ đạo</div>',
        unsafe_allow_html=True
    )

    hc1, hc2, hc3, hc4 = st.columns(4)
    with hc1:
        ha = st.select_slider(
            "Góc ném (°):", options=list(range(5, 86)),
            value=45, key="ha"
        )
    with hc2:
        hmin = st.number_input("h tối thiểu", 0.0, 10.0, 0.0, 0.1)
    with hc3:
        hmax_v = st.number_input("h tối đa", 0.1, 20.0, 2.0, 0.1)
    with hc4:
        hn = st.slider("Số bước", 3, 8, 6, 1)

    hvs = np.linspace(hmin, hmax_v, hn)
    fig5 = go.Figure()
    for i, hv in enumerate(hvs):
        c, d = COLORS[i % len(COLORS)], DASHES[i % len(DASHES)]
        if hv == 0:
            th = compute_no_drag(v0, ha, g, t_max)
            lb = "h = 0 (không cản)"
        else:
            th = compute_trajectory(x_fn, y_fn, vx_fn, vy_fn, m, g, hv, v0, ha, t_max)
            lb = f"h = {hv:.2f} kg/s"
        fig5.add_trace(go.Scatter(
            x=th['x'], y=th['y'], mode='lines',
            name=lb, line=dict(color=c, width=2.5, dash=d)
        ))

    fig5.update_layout(
        **LAYOUT, height=480,
        title=dict(
            text=f"Khảo sát hệ số cản (α = {ha}°, v₀ = {v0} m/s)",
            font=dict(size=15)
        ),
        xaxis_title="x (m)", yaxis_title="y (m)",
        xaxis=dict(rangemode='tozero', gridcolor='#f1f5f9'),
        yaxis=dict(rangemode='tozero', gridcolor='#f1f5f9'),
    )
    st.plotly_chart(fig5, width="stretch")


# ================================================================
# TAB 6: LÝ THUYẾT
# ================================================================
with tab6:
    st.markdown(
        '<div class="section-title">Cơ sở lý thuyết</div>',
        unsafe_allow_html=True
    )

    st.markdown("**Phương trình chuyển động (Newton II):**")
    st.latex(r"m\vec{a} = m\vec{g} - h\vec{v}")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Phương Ox (ngang):**")
        st.latex(r"m\frac{d^2x}{dt^2} + h\frac{dx}{dt} = 0")
    with col2:
        st.markdown("**Phương Oy (đứng):**")
        st.latex(r"m\frac{d^2y}{dt^2} + h\frac{dy}{dt} = -mg")

    st.markdown("**Điều kiện ban đầu:**")
    st.latex(
        r"x(0)=0,\; y(0)=0,\; "
        r"\dot{x}(0)=v_0\cos\alpha,\; "
        r"\dot{y}(0)=v_0\sin\alpha"
    )

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Nghiệm giải tích (SymPy)</div>',
        unsafe_allow_html=True
    )

    st.markdown("**Vị trí x(t):**")
    st.latex(f"x(t) = {x_ltx}")
    st.markdown("**Vị trí y(t):**")
    st.latex(f"y(t) = {y_ltx}")
    st.markdown("**Vận tốc vₓ(t):**")
    st.latex(f"v_x(t) = {vx_ltx}")
    st.markdown("**Vận tốc vᵧ(t):**")
    st.latex(f"v_y(t) = {vy_ltx}")

    st.markdown("---")
    st.markdown(
        '<div class="section-title">Nhận xét vật lý</div>',
        unsafe_allow_html=True
    )
    st.markdown("""
- **Vận tốc ngang** suy giảm theo hàm mũ e⁻ʰᵗ/ᵐ, tiến về 0 khi t → ∞.
- **Vận tốc đứng** tiến tới vận tốc giới hạn −mg/h (terminal velocity).
- **Quỹ đạo** không đối xứng: nhánh xuống dốc hơn nhánh lên.
- **Góc ném tối ưu** < 45° (khác trường hợp không cản).
- **Cơ năng** giảm dần do lực cản sinh công âm.
    """)
