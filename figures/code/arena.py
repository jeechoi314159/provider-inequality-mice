# -*- coding: utf-8 -*-
"""Colosseum arena 도식 (top view + side section).
규격 출처: Methods — 내측 원통 34 cm 지름 · 18 cm 높이, 게이트 6 × 10 cm 4개;
외측 원통 79 cm 지름 · 50 cm 높이; 로봇 14 × 11 × 7 cm."""
import numpy as np
from matplotlib.patches import Circle, Rectangle, Wedge, Ellipse, FancyBboxPatch
import figstyle as F

R_OUT, R_IN = 39.5, 17.0          # cm, 반지름
GATE_W, GATE_H = 6.0, 10.0        # cm
WALL = 2.4                        # 도식상 벽 두께 (cm)
GATE_DEG = np.degrees(GATE_W / R_IN)          # 게이트가 차지하는 각도
GATES = [40, 130, 220, 310]                   # 게이트 중심 각도


def _mouse(ax, x, y, ang=0, L=8.0, color='#6f6d68', zorder=5):
    """간단한 쥐 기호 (몸통 + 머리 + 꼬리), 길이 L cm."""
    t = np.radians(ang)
    ax.add_patch(Ellipse((x, y), L, L * 0.52, angle=ang, fc=color, ec='none', zorder=zorder))
    hx, hy = x + np.cos(t) * L * 0.46, y + np.sin(t) * L * 0.46
    ax.add_patch(Circle((hx, hy), L * 0.17, fc=color, ec='none', zorder=zorder))
    s = np.linspace(0, 1, 20)
    tx = x - np.cos(t) * (L * 0.5 + s * L * 0.75)
    ty = y - np.sin(t) * (L * 0.5 + s * L * 0.75) + np.sin(s * 3.1) * L * 0.12
    ax.plot(tx, ty, color=color, lw=0.6, zorder=zorder, solid_capstyle='round')


def _robot(ax, x, y):
    """스파이더 로봇 14 × 11 cm + 먹이."""
    for a in (28, 62, 118, 152, 208, 242, 298, 332):
        t = np.radians(a)
        ax.plot([x + np.cos(t) * 4.5, x + np.cos(t) * 10.5],
                [y + np.sin(t) * 4.0, y + np.sin(t) * 9.0],
                color=F.INK, lw=0.9, solid_capstyle='round', zorder=4)
    ax.add_patch(FancyBboxPatch((x - 6.0, y - 4.6), 12.0, 9.2,
                                boxstyle='round,pad=0.6,rounding_size=2.2',
                                fc='#3a3934', ec=F.INK, lw=0.6, zorder=5))
    ax.add_patch(Circle((x, y), 2.7, fc='#f2e2b8', ec=F.INK, lw=0.5, zorder=6))


def top_view(ax, labels=True):
    ax.set_aspect('equal'); ax.set_xlim(-R_OUT - 11, R_OUT + 11); ax.set_ylim(-R_OUT - 14, R_OUT + 7)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    # 외측 원통 (거주 구역)
    ax.add_patch(Circle((0, 0), R_OUT, fc='#faf9f7', ec=F.INK, lw=1.1, zorder=0))

    # 내측 원통 벽: 두꺼운 고리에서 게이트만 뚫음 → 갇힌 구조가 보이도록
    for k, g in enumerate(GATES):
        a0 = g + GATE_DEG / 2
        a1 = GATES[(k + 1) % 4] - GATE_DEG / 2
        if a1 < a0: a1 += 360
        ax.add_patch(Wedge((0, 0), R_IN + WALL, a0, a1, width=WALL,
                           fc='#6b6963', ec=F.INK, lw=0.5, zorder=3))
    ax.add_patch(Circle((0, 0), R_IN, fc='#ffffff', ec='none', zorder=1))

    # 게이트: 양쪽 문설주와 개구부 표시
    for g in GATES:
        for s in (-1, 1):
            a = np.radians(g + s * GATE_DEG / 2)
            ax.plot([np.cos(a) * (R_IN - 0.3), np.cos(a) * (R_IN + WALL + 0.3)],
                    [np.sin(a) * (R_IN - 0.3), np.sin(a) * (R_IN + WALL + 0.3)],
                    color=F.PROV, lw=1.0, solid_capstyle='round', zorder=6)
        aa = np.radians(np.linspace(g - GATE_DEG / 2, g + GATE_DEG / 2, 12))
        ax.plot(np.cos(aa) * (R_IN + WALL / 2), np.sin(aa) * (R_IN + WALL / 2),
                color=F.PROV, lw=1.0, zorder=6)

    _robot(ax, 2.0, 3.0)
    _mouse(ax, -7.0, -8.0, ang=55, L=6.2, color=F.PROV)                 # 안쪽 1마리 (제공자)
    for x, y, a in [(-30, 3, -15), (-21, -25, 30), (11, -31, 75),
                    (22, -22, 140), (27, 13, 205)]:                     # 바깥 5마리
        _mouse(ax, x, y, ang=a, L=7.0)

    if labels:
        lab = dict(fontsize=6.5, color=F.INK, va='center')
        arr = lambda c: dict(arrowstyle='->', lw=0.6, color=c, shrinkA=1, shrinkB=1)
        ax.annotate('Robot zone\ndiameter = 34 cm', xy=(9.0, 12.5), xytext=(34, 32),
                    ha='center', arrowprops=arr(F.INK), **lab)
        ax.annotate('Living zone\ndiameter = 79 cm', xy=(-30, 17), xytext=(-34, 41),
                    ha='center', arrowprops=arr(F.INK), **lab)
        g = np.radians(GATES[3])
        ax.annotate('Gate 6 × 10 cm\n(Mouse only)',
                    xy=(np.cos(g) * (R_IN + WALL / 2), np.sin(g) * (R_IN + WALL / 2)),
                    xytext=(30, -47), ha='center', arrowprops=arr(F.PROV), **lab)
        ax.annotate('Robot with snack', xy=(-5.5, -1.0), xytext=(-30, -47),
                    ha='center', arrowprops=arr(F.INK), **lab)


def side_view(ax, compact=True):
    """단면: 내측 벽 18 cm, 외측 벽 50 cm, 게이트 10 cm — 로봇은 나올 수 없음."""
    ax.set_aspect('equal'); ax.set_xlim(-R_OUT - 5, R_OUT + 5); ax.set_ylim(-13, 60)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.plot([-R_OUT, R_OUT], [0, 0], color=F.INK, lw=0.9)
    for x in (-R_OUT, R_OUT):
        ax.add_patch(Rectangle((x - 1.1, 0), 2.2, 50, fc='#6b6963', ec='none'))
    for x in (-R_IN, R_IN):
        ax.add_patch(Rectangle((x - 1.1, GATE_H), 2.2, 18 - GATE_H, fc='#6b6963', ec='none'))
        ax.plot([x, x], [0, GATE_H], color=F.PROV, lw=1.8, solid_capstyle='butt')
    for sgn in (-1, 1):
        for dx, kx, kh, fx in ((4.5, 7.5, 9.0, 9.5), (5.5, 9.5, 7.0, 11.5)):
            ax.plot([sgn * dx, sgn * kx, sgn * fx], [5.0, kh, 0.0],
                    color=F.INK, lw=0.7, solid_capstyle='round', solid_joinstyle='round', zorder=2)
    ax.add_patch(FancyBboxPatch((-6.0, 0.4), 12.0, 6.6, boxstyle='round,pad=0.4,rounding_size=1.4',
                                fc='#3a3934', ec=F.INK, lw=0.5, zorder=3))
    ax.add_patch(Circle((0, 7.6), 2.2, fc='#f2e2b8', ec=F.INK, lw=0.5, zorder=4))
    _mouse(ax, 29, 3.2, ang=180, L=7.0)
    ax.text(R_OUT + 1.5, 50, '50 cm', fontsize=5.8, color=F.INK, ha='right', va='bottom')
    ax.text(R_IN + 2.5, 18.5, '18 cm', fontsize=5.8, color=F.INK, ha='left', va='bottom')
    ax.text(-R_IN - 2.5, GATE_H + 0.5, '10 cm', fontsize=5.8, color=F.PROV, ha='right', va='bottom')
    ax.annotate('', xy=(-R_IN + 1.5, 2.5), xytext=(-R_IN - 8.0, 2.5),
                arrowprops=dict(arrowstyle='->', lw=0.6, color=F.PROV))
    ax.text(0, 56, 'Side section', fontsize=5.8, color=F.INK2, ha='center', va='center')
    ax.text(0, -7.0, 'Robot confined; mice pass the gates',
            fontsize=5.8, color=F.INK2, ha='center', va='top')


def task_sequence(ax):
    """Entry → Ride → Get → Carry out → Eat, top_view 와 같은 글리프."""
    STEPS = ['Entry', 'Ride', 'Get', 'Carry out', 'Eat']
    N = len(STEPS); Wd, Hd = 100.0, 74.0
    ax.set_aspect('equal'); ax.set_xlim(-46, N * Wd + 2); ax.set_ylim(-4, Hd + 8)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)

    def wall(x0, gate_open=True):
        """가로 벽 (위쪽이 robot zone), 가운데 게이트."""
        ax.plot([x0 + 6, x0 + 40], [34, 34], color='#6b6963', lw=3, solid_capstyle='butt')
        ax.plot([x0 + 54, x0 + 88], [34, 34], color='#6b6963', lw=3, solid_capstyle='butt')
        ax.plot([x0 + 40, x0 + 40], [31, 37], color=F.PROV, lw=1.0)
        ax.plot([x0 + 54, x0 + 54], [31, 37], color=F.PROV, lw=1.0)

    def bot(x0, x, y, sc=1.0):
        # 무릎이 있는 다리 → 물체가 아니라 보행 로봇으로 보이게
        for sgn in (-1, 1):
            for dx, kx, kh, fx in ((7, 12, 7, 15), (8, 15, 5, 19), (7, 17, 2, 22)):
                ax.plot([x0 + x + sgn * dx * sc, x0 + x + sgn * kx * sc, x0 + x + sgn * fx * sc],
                        [y + 2 * sc, y + kh * sc, y - 7 * sc],
                        color=F.INK, lw=0.8, solid_capstyle='round',
                        solid_joinstyle='round', zorder=3)
        ax.add_patch(FancyBboxPatch((x0 + x - 9 * sc, y - 6 * sc), 18 * sc, 12 * sc,
                                    boxstyle='round,pad=0.8,rounding_size=2.5',
                                    fc='#3a3934', ec=F.INK, lw=0.5, zorder=4))

    def snack(x, y, r=3.2):
        ax.add_patch(Circle((x, y), r, fc='#f2e2b8', ec=F.INK, lw=0.5, zorder=6))

    for k, (name, x0) in enumerate(zip(STEPS, np.arange(N) * Wd)):
        ax.text(x0 + 47, Hd + 1, name, fontsize=7, color=F.INK, ha='center', va='bottom')
        wall(x0)
        if k == 0:                                   # Entry
            bot(x0, 47, 54); snack(x0 + 47, 54)
            _mouse(ax, x0 + 47, 24, ang=90, L=13, color=F.PROV)
        elif k == 1:                                 # Ride
            bot(x0, 47, 52); snack(x0 + 47, 52)
            _mouse(ax, x0 + 47, 46, ang=90, L=12, color=F.PROV)
        elif k == 2:                                 # Get
            bot(x0, 38, 52)
            _mouse(ax, x0 + 60, 50, ang=-30, L=12, color=F.PROV); snack(x0 + 68, 45)
        elif k == 3:                                 # Carry out
            bot(x0, 30, 54)
            _mouse(ax, x0 + 47, 20, ang=-90, L=13, color=F.PROV); snack(x0 + 47, 11)
        else:                                        # Eat
            snack(x0 + 47, 16, 4.0)
            for a in (10, 80, 150, 220, 290):
                tt = np.radians(a)
                _mouse(ax, x0 + 47 + np.cos(tt) * 15, 16 + np.sin(tt) * 12,
                       ang=a + 180, L=11, color=F.PROV if a == 80 else F.OTH)
        if k < N - 1:                                # 단계 사이 가운데에 진행 방향 표시
            ax.annotate('', xy=(x0 + 103, Hd / 2), xytext=(x0 + 91, Hd / 2),
                        arrowprops=dict(arrowstyle='-|>', lw=1.1, color='#8d8b85',
                                        mutation_scale=9), zorder=2)
    ax.text(-44, 50, 'Robot zone', fontsize=6.5, color=F.INK, ha='left', va='center')
    ax.text(-44, 18, 'Living zone', fontsize=6.5, color=F.INK, ha='left', va='center')
    ax.annotate('', xy=(-6, 34), xytext=(-20, 34),
                arrowprops=dict(arrowstyle='-', lw=0.6, color=F.INK))


def fixed_snack(ax):
    """고정 먹이 대조 삽화: 멈춘 로봇에 먹이가 붙어 있고 쥐들이 몰려듦."""
    ax.set_aspect('equal'); ax.set_xlim(-26, 26); ax.set_ylim(-24, 30)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    ax.add_patch(Circle((0, -1), 22, fc='#faf9f7', ec=F.INK, lw=0.8, zorder=0))
    for sgn in (-1, 1):                                   # 멈춘 다리
        for dx, kx, kh, fx in ((5, 9, 5, 12), (6, 11, 2, 15)):
            ax.plot([sgn * dx, sgn * kx, sgn * fx], [0.5, kh, -7],
                    color=F.INK, lw=0.7, solid_capstyle='round',
                    solid_joinstyle='round', zorder=2)
    ax.add_patch(FancyBboxPatch((-6.5, -5.0), 13.0, 7.5,
                                boxstyle='round,pad=0.5,rounding_size=1.6',
                                fc='#3a3934', ec=F.INK, lw=0.5, zorder=3))
    ax.add_patch(Circle((0, 3.6), 2.6, fc='#f2e2b8', ec=F.INK, lw=0.5, zorder=4))
    for a, col in ((200, F.OTH), (265, F.PROV), (330, F.OTH)):
        t = np.radians(a)
        _mouse(ax, np.cos(t) * 13.0, -1 + np.sin(t) * 11.5, ang=a + 180, L=8.5, color=col)
    ax.annotate('Snack fixed', xy=(0, 6.4), xytext=(0, 26), fontsize=5.8,
                color=F.INK, ha='center', va='center',
                arrowprops=dict(arrowstyle='->', lw=0.6, color=F.PROV, shrinkA=2, shrinkB=2))


def chamber(ax):
    """닫힌 칸 삽화: robot zone 게이트를 나오면 바로 있는 칸, 한 방향 문, 밖의 나머지."""
    ax.set_aspect('equal'); ax.set_xlim(-40, 30); ax.set_ylim(-34, 30)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values(): s_.set_visible(False)
    CX, CY, RR = -95.0, 0.0, 78.0                            # robot zone 벽 (가운데가 게이트)
    for a0, a1 in ((-16, -7), (7, 16)):
        t = np.radians(np.linspace(a0, a1, 24))
        ax.plot(CX + np.cos(t) * RR, CY + np.sin(t) * RR,
                color='#6b6963', lw=2.2, solid_capstyle='butt', zorder=0)
    for sgn in (-1, 1):                                       # 게이트 문설주
        t = np.radians(7 * sgn)
        ax.plot([CX + np.cos(t) * (RR - 1.6), CX + np.cos(t) * (RR + 1.6)],
                [CY + np.sin(t) * (RR - 1.6), CY + np.sin(t) * (RR + 1.6)],
                color=F.PROV, lw=1.0, solid_capstyle='round', zorder=2)
    ax.text(-25, 25, 'Robot zone', fontsize=5.8, color=F.INK, ha='center', va='center')
    ax.add_patch(Rectangle((-14, -9), 30, 18, fc='#faf9f7', ec=F.INK, lw=0.8, zorder=1))
    ax.plot([16, 16], [-9, -3], color='#ffffff', lw=2.4, zorder=2)      # 문 틈
    ax.plot([16, 22], [-3, -7], color=F.PROV, lw=1.0, zorder=3)         # 한 방향 문
    _mouse(ax, -1, 0, ang=15, L=9.5, color=F.PROV)
    ax.add_patch(Circle((5.5, 2.0), 2.5, fc='#f2e2b8', ec=F.INK, lw=0.5, zorder=6))
    for x, y, a in ((22, -17, 140), (3, -20, 75), (-8, -17, 25)):
        _mouse(ax, x, y, ang=a, L=8.5, color=F.OTH)
    ax.annotate('One-way door', xy=(19.0, -5.5), xytext=(-2, -30), fontsize=5.8,
                color=F.INK, ha='center', va='center',
                arrowprops=dict(arrowstyle='->', lw=0.6, color=F.PROV, shrinkA=2, shrinkB=2))
