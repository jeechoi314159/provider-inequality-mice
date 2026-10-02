# -*- coding: utf-8 -*-
"""Science 규격 그림 스타일. 모든 build 스크립트가 이 모듈만 통해 그림을 연다.

Science (initial manuscript) 지침 반영:
 - 폭 5.7 / 12.1 / 18.4 cm, sans-serif(Helvetica 우선)
 - 글자 7 pt 목표·5 pt 최소, 선 0.5 pt 이상, 기호 6 pt 이상
 - 패널 문자 대문자 10 pt 굵게 좌상단, 그림 안 제목/소제목 없음
 - 축은 변수명 + 단위 괄호, 첫 글자만 대문자, minor tick·격자 없음
 - 적록 병용 금지
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.font_manager as fm

# submission/fonts/ 에 글꼴 파일(.ttf/.otf)을 넣어두면 자동 등록됨.
# Science 규정 글꼴(Arial)을 쓰려면 Arial 4종(정체·굵게·기울임·굵은기울임)을 이 폴더에 둘 것.
# 같은 family 를 주장하는 Narrow·Black·Light·Medium·부분 글꼴이 섞이면 잘못 선택될 수 있으므로
# 네 가지 표준 스타일이면서 글리프가 충분한 파일만 등록한다.
_FONTDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fonts')
_OKSTYLE = {'Regular', 'Bold', 'Italic', 'Bold Italic'}
if os.path.isdir(_FONTDIR):
    from matplotlib import ft2font as _ft
    for _f in sorted(os.listdir(_FONTDIR)):
        if not _f.lower().endswith(('.ttf', '.otf')):
            continue
        _p = os.path.join(_FONTDIR, _f)
        try:
            _face = _ft.FT2Font(_p)
            if _face.style_name not in _OKSTYLE or _face.num_glyphs < 1000:
                continue
            fm.fontManager.addfont(_p)
        except Exception:
            pass

CM = 1 / 2.54
W1, W2, W3 = 5.7 * CM, 12.1 * CM, 18.4 * CM          # 1·2·3단 폭 (inch)

# 그림 안 모든 글자는 Arial. Arial 이 없는 환경에서는 metric 호환 글꼴로 대체됨.
_PREF = ['Arial', 'Helvetica', 'Liberation Sans', 'Nimbus Sans', 'DejaVu Sans']
_HAVE = {f.name for f in fm.fontManager.ttflist}
FONT = next((f for f in _PREF if f in _HAVE), 'DejaVu Sans')

# provider 주황 / others 회색 / 진입 전 γ 파랑 / 진입 후 β 보라 (적록 병용 없음)
PROV, OTH, OTHL, GRID = '#e2661f', '#8d8b85', '#c9c7c0', '#d9d7d2'
INK = INK2 = '#000000'      # 그림 안 모든 글자는 검정
GAM, BET = '#2a78d6', '#5b3fa8'
ROLE = {'retrieved': PROV, 'entered': OTHL, 'stayed out': INK2}

rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': [FONT], 'font.size': 7,
    'mathtext.fontset': 'custom', 'mathtext.rm': FONT, 'mathtext.it': FONT,
    'mathtext.bf': FONT, 'mathtext.sf': FONT, 'mathtext.default': 'regular',
    'axes.labelsize': 8, 'axes.titlesize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'legend.fontsize': 7, 'figure.titlesize': 8,
    'axes.linewidth': 0.5, 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
    'xtick.major.size': 2.2, 'ytick.major.size': 2.2,
    'xtick.minor.size': 0, 'ytick.minor.size': 0,
    'lines.linewidth': 0.8, 'lines.markersize': 3,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'axes.axisbelow': True,
    'legend.frameon': False, 'legend.labelcolor': INK, 'legend.handlelength': 1.2, 'legend.handletextpad': 0.5,
    'savefig.dpi': 600, 'figure.dpi': 200, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.unicode_minus': False, 'text.color': INK,
    'mathtext.fontset': 'custom', 'mathtext.rm': FONT,
    'mathtext.it': FONT + ':italic', 'mathtext.bf': FONT + ':bold',
    'mathtext.cal': FONT, 'mathtext.sf': FONT, 'mathtext.tt': FONT,
    'mathtext.default': 'it',
    'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK, 'axes.edgecolor': INK,
})


def figure(width=W3, height=6.0 * CM):
    """새 그림. width 는 W1/W2/W3 중 하나."""
    return plt.figure(figsize=(width, height))


def panel(fig, rect):
    """rect = (left, bottom, w, h), 0–1 비율."""
    return fig.add_axes(rect)


def label(fig, letter, x, y):
    """패널 문자: 대문자 10 pt 굵게, 좌상단."""
    fig.text(x, y, letter, fontsize=10, fontweight='bold', va='top', ha='left', color=INK)


def tidy(ax, xlab=None, ylab=None):
    if xlab: ax.set_xlabel(xlab)
    if ylab: ax.set_ylabel(ylab)
    ax.tick_params(which='minor', length=0)
    return ax


def note(ax, text, x=0.98, y=0.98, ha='right', va='top', size=7.0, color=INK):
    """패널 안 최소 표기(핵심 통계 하나). 제목이 아니라 통계값만."""
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=color)


def pending(ax, text='자료 확보 예정'):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_linestyle((0, (2, 2))); s.set_color(OTH); s.set_visible(True)
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha='center', va='center',
            fontsize=7.0, color=OTH)


def save(fig, stem, outdir_png, outdir_pdf):
    import os
    if os.environ.get('AUDIT'):
        try:
            import audit as _A
            _t, _n = _A.audit(fig, stem, os.path.join(os.path.dirname(outdir_png), 'audit'))
            print(_t)
        except Exception as _e:
            print('audit failed', stem, _e)
    fig.savefig(os.path.join(outdir_png, stem + '.png'), dpi=600,
                facecolor='white', bbox_inches=None)
    fig.savefig(os.path.join(outdir_pdf, stem + '.pdf'),
                facecolor='white', bbox_inches=None)
    plt.close(fig)
    return stem


# ===================================================================== 공통 격자
class Grid(object):
    """cm 단위 공통 격자.

    모든 그림이 같은 바깥 여백·같은 패널 간격을 쓰도록 하고, 셀 안에서 축 영역을
    고정된 패딩만큼 들여 놓아 (1) 같은 열의 패널은 왼쪽 기준선이, (2) 같은 행의
    패널은 위·아래 기준선이 자동으로 맞도록 한다. 축 라벨·눈금은 그 패딩 안에
    들어가므로 이웃 셀을 침범하지 않는다.
    """

    def __init__(self, fig, nrow, ncol, left=0.80, right=0.25, top=0.50, bottom=0.30,
                 wgap=0.85, hgap=0.95, wratios=None, hratios=None):
        self.fig = fig
        self.W = fig.get_figwidth() * 2.54
        self.H = fig.get_figheight() * 2.54
        self.nrow, self.ncol = nrow, ncol
        self.left, self.right, self.top, self.bottom = left, right, top, bottom
        self.wgap, self.hgap = wgap, hgap
        wr = list(wratios) if wratios else [1.0] * ncol
        hr = list(hratios) if hratios else [1.0] * nrow
        availw = self.W - left - right - wgap * (ncol - 1)
        availh = self.H - top - bottom - hgap * (nrow - 1)
        sw, sh = float(sum(wr)), float(sum(hr))
        self.cw = [availw * v / sw for v in wr]
        self.ch = [availh * v / sh for v in hr]
        self.x0 = [left + sum(self.cw[:c]) + wgap * c for c in range(ncol)]
        # 행은 위에서부터 0번
        self.ytop = [self.H - top - sum(self.ch[:r]) - hgap * r for r in range(nrow)]

    # ----------------------------------------------------------- 셀 (cm → 비율)
    def cell_cm(self, r, c, rs=1, cs=1):
        x = self.x0[c]
        w = sum(self.cw[c:c + cs]) + self.wgap * (cs - 1)
        ytop = self.ytop[r]
        h = sum(self.ch[r:r + rs]) + self.hgap * (rs - 1)
        return x, ytop - h, w, h

    def cell(self, r, c, rs=1, cs=1):
        x, y, w, h = self.cell_cm(r, c, rs, cs)
        return [x / self.W, y / self.H, w / self.W, h / self.H]

    # ------------------------------------------------------------------- 축
    def ax(self, r, c, padl=1.05, padb=0.90, padr=0.05, padt=0.05, rs=1, cs=1):
        """셀 안에 축을 놓는다. pad 는 cm, 축 라벨·눈금이 들어갈 자리."""
        x, y, w, h = self.cell_cm(r, c, rs, cs)
        ax = self.fig.add_axes([(x + padl) / self.W, (y + padb) / self.H,
                                (w - padl - padr) / self.W, (h - padb - padt) / self.H],
                               label='r%dc%d' % (r, c))
        ax.tick_params(which='minor', length=0)
        return ax

    def rect(self, r, c, padl=0.0, padb=0.0, padr=0.0, padt=0.0, rs=1, cs=1):
        x, y, w, h = self.cell_cm(r, c, rs, cs)
        return [(x + padl) / self.W, (y + padb) / self.H,
                (w - padl - padr) / self.W, (h - padb - padt) / self.H]

    # -------------------------------------------------- 패널 문자 (셀 좌상단)
    def label(self, r, c, letter, dx=0.0, dy=0.0, rs=1, cs=1):
        x, y, w, h = self.cell_cm(r, c, rs, cs)
        label(self.fig, letter, (x + dx) / self.W, (y + h + 0.28 + dy) / self.H)

    def fx(self, cm):
        return cm / self.W

    def fy(self, cm):
        return cm / self.H
