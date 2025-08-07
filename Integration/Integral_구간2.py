import cv2
import numpy as np
import matplotlib.pyplot as plt

# ────────────── 옵션 ──────────────
SHOW_MODE   = "offset"      # "offset" / "alpha"
OFFSET_STEP = 200000
ALPHA       = 0.25
COLORMAP    = plt.cm.viridis
FILL_COL_1  = "#39d07d"     # 첫 번째 구간 채움
FILL_COL_2  = "#ff9933"     # 두 번째   "
LINE_COL    = "#555555"     # 비율 연결선
# ────────────────────────────────

# 0. 이미지 ---------------------------------------------------------------
IMAGE_PATH = "original_Mono12_20250411_110738.tiff"
orig = cv2.imread(IMAGE_PATH, cv2.IMREAD_UNCHANGED)
if orig is None:
    raise FileNotFoundError("이미지를 불러올 수 없습니다.")
disp_base = ((orig/4095)*255).astype(np.uint8) if orig.dtype==np.uint16 else orig.copy()

# 1. 상·하 30 px 제거 ------------------------------------------------------
TRIM_T, TRIM_B = 30, 30
trimmed = orig[TRIM_T : orig.shape[0]-TRIM_B]

# 2. 프리뷰 축소 -----------------------------------------------------------
def fit(img, mw=1280, mh=720):
    h,w = img.shape[:2]; s = min(mw/w, mh/h)
    return (cv2.resize(img,(int(w*s),int(h*s)),cv2.INTER_AREA), s) if s<1 else (img.copy(),1)
disp_img, disp_scale = fit(disp_base)
sel_rows = []

# 3. nm x축 ---------------------------------------------------------------
x_nm = np.linspace(950, 750, orig.shape[1])

# 4. abs(dx) 적분 ----------------------------------------------------------
def pure_peak_area(x_sel, y_sel):
    m = (y_sel[-1]-y_sel[0]) / (x_sel[-1]-x_sel[0])
    b = y_sel[0] - m*x_sel[0]
    y_line = m*x_sel + b
    area_curve = area_base = 0.0
    for i in range(len(x_sel)-1):
        dx = abs(x_sel[i+1] - x_sel[i])
        area_curve += 0.5*(y_sel[i]+y_sel[i+1])*dx
        area_base  += 0.5*(y_line[i]+y_line[i+1])*dx
    return area_curve - area_base, y_line

# 5. 층층이 + 비율 그래프 ---------------------------------------------------
def stacked_plot(idx_pairs):
    count = 7
    seg_h = int(len(trimmed)/count)
    # fig  = plt.figure(figsize=(16,8))
    fig  = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs   = fig.add_gridspec(1,2,width_ratios=[2,1],wspace=0.05)
    ax   = fig.add_subplot(gs[0,0])
    axr  = fig.add_subplot(gs[0,1], sharey=ax)

    ratios, y_coords, colors = [], [], []

    for plot_idx, i in enumerate(reversed(range(count))):
        r0, r1 = i*seg_h, (i+1)*seg_h
        seg_int = np.sum(trimmed[r0:r1], axis=0).astype(float)

        ofs = plot_idx * OFFSET_STEP if SHOW_MODE == "offset" else 0.0
        y_plot = seg_int + ofs
        color  = COLORMAP(i/(count-1))

        # 곡선
        ax.plot(x_nm, y_plot, color=color, lw=0.8,
                alpha=1.0 if SHOW_MODE=="offset" else ALPHA)

        # 두 구간 적분 & 채우기
        areas = []
        for p, (a,b) in enumerate(idx_pairs):
            x_sel = x_nm[a:b+1];  y_sel = seg_int[a:b+1]
            pure, y_line = pure_peak_area(x_sel, y_sel)
            areas.append(pure)
            fill = FILL_COL_1 if p==0 else FILL_COL_2
            ax.fill_between(x_sel, y_line+ofs, y_sel+ofs,
                            color=fill,
                            alpha=0.4 if SHOW_MODE=="offset" else ALPHA*1.6)

        ratio   = areas[1] / areas[0] if areas[0] else np.nan
        y_right = y_plot[-1]               # 750 nm 위치로 정렬

        ratios .append(ratio)
        y_coords.append(y_right)
        colors .append(color)

        # 라벨 (세그 번호 & 두 넓이)
        ax.text(750, y_right + 10000,
                f"{i:02d}:{int(areas[0]):,}/{int(areas[1]):,}",
                ha="right", va="center", fontsize=8,
                color=color if SHOW_MODE=="offset" else "#333")

    # 비율 그래프: 선 + 점 (점은 곡선 색)
    axr.plot(ratios, y_coords, '-', color=LINE_COL, lw=1.2)
    for r, y, c in zip(ratios, y_coords, colors):
        axr.plot(r, y, 'o', color=c, markersize=6)

    # 축 & 레이아웃
    ax.invert_xaxis()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity" + (" (offset)" if SHOW_MODE=="offset" else ""))
    ax.set_title("${count} Segments")

    axr.axvline(1.0, ls="--", color="#888", lw=0.8)
    axr.set_xlabel("Area₂ / Area₁")
    axr.set_xlim(left=0)
    axr.grid(axis='x', ls=':', alpha=0.4)
    plt.setp(axr.get_yticklabels(), visible=False)

    plt.show(block=False)

# 6. 네 점 클릭 (x1,x2,x3,x4) ---------------------------------------------
def pick_four(x, y):
    fig, ax = plt.subplots(figsize=(12,7))
    ax.plot(x, y, c="blue"); ax.invert_xaxis()
    ax.set_title("x₁, x₂, x₃, x₄")
    clicked = []
    near = lambda arr,v: np.abs(arr - v).argmin()
    def onclick(ev):
        if ev.inaxes is None or ev.button!=1: return
        clicked.append(ev.xdata)
        ax.plot(ev.xdata, ev.ydata, 'ro'); fig.canvas.draw()
        if len(clicked)==4:
            idx = [near(x, v) for v in clicked]
            idx_pairs = [tuple(sorted(idx[:2])), tuple(sorted(idx[2:]))]
            plt.close(fig); stacked_plot(idx_pairs)
    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

# 7. 행 두 곳 클릭 ---------------------------------------------------------
def draw_rows():
    tmp = cv2.cvtColor(disp_img.copy(), cv2.COLOR_GRAY2BGR)
    for r in sel_rows:
        cv2.line(tmp, (0,int(r*disp_scale)),
                 (tmp.shape[1],int(r*disp_scale)), (0,255,0), 2)
    cv2.imshow("Select Rows", tmp)

def mouse_cb(evt,x,y,flags,param):
    if evt==cv2.EVENT_LBUTTONDOWN:
        row = int(y/disp_scale)
        if len(sel_rows)==2: sel_rows.clear()
        sel_rows.append(row); draw_rows()
        if len(sel_rows)==2:
            s,e = sorted(sel_rows)
            intensity = np.sum(orig[s:e], axis=0).astype(float)
            pick_four(x_nm, intensity)
            sel_rows.clear(); draw_rows()

cv2.namedWindow("Select Rows")
cv2.setMouseCallback("Select Rows", mouse_cb)
draw_rows()
print("행 두 곳 클릭 → 그래프 네 점 클릭 → 층층이 + 비율 (정렬 라벨)\nESC 로 종료")

while cv2.waitKey(20)!=27: pass
cv2.destroyAllWindows()
