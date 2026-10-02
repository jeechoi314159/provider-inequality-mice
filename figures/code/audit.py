# -*- coding: utf-8 -*-
"""렌더링된 그림의 bounding box 검수.

figstyle.save() 가 AUDIT=1 일 때 호출한다. 검사 항목
  1) 캔버스 밖으로 나간 요소(잘림)
  2) 글자끼리 겹침
  3) 패널(축 tight bbox)끼리 겹침
  4) 같은 행/열 패널의 기준선 어긋남
  5) 패널 사이 간격 불균등
자동 검사는 시각 검수를 대체하지 않는다(글자-그래픽 겹침은 눈으로 봐야 한다).
"""
import os
CM = 2.54
TOL = 0.015          # cm, 겹침 판정 시 각 상자를 이만큼 줄인다
ALIGN_TOL = 0.03     # cm, 기준선 어긋남 허용
GAP_TOL = 0.30     # cm, 같은 위계 간격 편차 허용(라벨 길이 차이로 ink 간격은 완전히 같을 수 없음)


def _live_ticklabels(ax, which):
    """보이는 범위 안에 있는 눈금 라벨만. (범위 밖 라벨은 그려지지 않는다)"""
    axis = ax.xaxis if which == 'x' else ax.yaxis
    lo, hi = (ax.get_xlim() if which == 'x' else ax.get_ylim())
    lo, hi = min(lo, hi), max(lo, hi)
    locs = axis.get_ticklocs()
    labs = axis.get_ticklabels()
    out = []
    for loc, lab in zip(locs, labs):
        if lo - 1e-9 <= loc <= hi + 1e-9:
            out.append(lab)
    return out


def _texts(ax):
    out = []
    for t in list(ax.texts):
        out.append(t)
    for t in (ax.title, ax.xaxis.label, ax.yaxis.label):
        out.append(t)
    out += _live_ticklabels(ax, 'x') + _live_ticklabels(ax, 'y')
    return out


def collect(fig):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    dpi = float(fig.dpi)
    W, H = fig.get_figwidth() * CM, fig.get_figheight() * CM

    def cm(bb):
        return [bb.x0 / dpi * CM, bb.y0 / dpi * CM, bb.x1 / dpi * CM, bb.y1 / dpi * CM]

    texts, panels = [], []
    for t in list(fig.texts):
        if t.get_visible() and t.get_text().strip():
            texts.append(('figure', 'fig-text', t.get_text(), cm(t.get_window_extent(rend))))
    for li, lg in enumerate(fig.legends):
        for t in lg.get_texts():
            texts.append(('figure', 'fig-legend%d' % li, t.get_text(),
                          cm(t.get_window_extent(rend))))
    for ai, ax in enumerate(fig.axes):
        nm = 'ax%d' % ai
        if ax.get_label():
            nm += '(' + str(ax.get_label()) + ')'
        try:
            panels.append((nm, cm(ax.get_tightbbox(rend)), cm(ax.get_window_extent(rend))))
        except Exception:
            continue
        for t in _texts(ax):
            if t.get_visible() and t.get_text().strip():
                try:
                    texts.append((nm, 'text', t.get_text(), cm(t.get_window_extent(rend))))
                except Exception:
                    pass
        lg = ax.get_legend()
        if lg is not None and lg.get_visible():
            for t in lg.get_texts():
                texts.append((nm, 'legend', t.get_text(), cm(t.get_window_extent(rend))))
    return W, H, texts, panels


def _ov(a, b, tol=TOL):
    dx = min(a[2], b[2]) - max(a[0], b[0]) - 2 * tol
    dy = min(a[3], b[3]) - max(a[1], b[1]) - 2 * tol
    return (dx, dy) if (dx > 0 and dy > 0) else None


def _inside(a, b):
    return a[0] >= b[0] - 0.02 and a[1] >= b[1] - 0.02 and a[2] <= b[2] + 0.02 and a[3] <= b[3] + 0.02


def audit(fig, stem, outdir=None):
    W, H, texts, panels = collect(fig)
    P = []
    P.append('%s  %.1f x %.1f cm  패널 %d' % (stem, W, H, len(panels)))

    # 1) 캔버스 밖
    for own, kind, s, b in texts:
        if b[0] < -0.005 or b[1] < -0.005 or b[2] > W + 0.005 or b[3] > H + 0.005:
            P.append('  [잘림] %s %s %r  x %.2f–%.2f  y %.2f–%.2f'
                     % (own, kind, s[:34], b[0], b[2], b[1], b[3]))
    for nm, tb, _ in panels:
        if tb[0] < -0.005 or tb[1] < -0.005 or tb[2] > W + 0.005 or tb[3] > H + 0.005:
            P.append('  [잘림] %s 패널 전체  x %.2f–%.2f  y %.2f–%.2f' % (nm, tb[0], tb[2], tb[1], tb[3]))

    # 2) 글자끼리
    for i in range(len(texts)):
        oi, ki, si, bi = texts[i]
        for j in range(i + 1, len(texts)):
            oj, kj, sj, bj = texts[j]
            if oi == oj and ki == kj and ki in ('legend', 'fig-legend0', 'fig-legend1'):
                continue
            o = _ov(bi, bj)
            if o:
                P.append('  [글자겹침] %s%r × %s%r  %.2f×%.2f cm'
                         % (oi, si[:24], oj, sj[:24], o[0], o[1]))

    # 3) 패널끼리
    for i in range(len(panels)):
        ni, ti, ri = panels[i]
        for j in range(i + 1, len(panels)):
            nj, tj, rj = panels[j]
            if _inside(rj, ri) or _inside(ri, rj):      # 삽입 그림
                continue
            o = _ov(ti, tj, 0.0)
            if o and o[0] > 0.02 and o[1] > 0.02:
                P.append('  [패널겹침] %s × %s  %.2f×%.2f cm' % (ni, nj, o[0], o[1]))

    # 4) 기준선  (label 이 '<sub' 로 시작하는 축 = 한 패널 안의 하위 격자, 정렬·간격 검사 제외)
    rows, cols = {}, {}
    for nm, tb, rb in panels:
        if '(<sub' in nm:
            continue
        rows.setdefault(round(rb[3], 1), []).append((nm, rb))
        cols.setdefault(round(rb[0], 1), []).append((nm, rb))
    for k, v in sorted(rows.items()):
        if len(v) < 2:
            continue
        tops = [b[3] for _, b in v]; bots = [b[1] for _, b in v]
        if max(tops) - min(tops) > ALIGN_TOL:
            P.append('  [정렬] 같은 행 상단 어긋남 %.2f cm: %s'
                     % (max(tops) - min(tops), ', '.join(n for n, _ in v)))
        if max(bots) - min(bots) > ALIGN_TOL:
            P.append('  [정렬] 같은 행 하단 어긋남 %.2f cm: %s'
                     % (max(bots) - min(bots), ', '.join(n for n, _ in v)))
    for k, v in sorted(cols.items()):
        if len(v) < 2:
            continue
        ls = [b[0] for _, b in v]
        if max(ls) - min(ls) > ALIGN_TOL:
            P.append('  [정렬] 같은 열 좌측 어긋남 %.2f cm: %s'
                     % (max(ls) - min(ls), ', '.join(n for n, _ in v)))

    # 5) 행 안 간격
    for k, v in sorted(rows.items()):
        if len(v) < 3:
            continue
        v = sorted(v, key=lambda t: t[1][0])
        gaps = [v[i + 1][1][0] - v[i][1][2] for i in range(len(v) - 1)]
        if max(gaps) - min(gaps) > GAP_TOL:
            P.append('  [간격] 행 간격 불균등 %s cm: %s'
                     % (' / '.join('%.2f' % g for g in gaps), ', '.join(n for n, _ in v)))

    bad = [l for l in P[1:]]
    P.append('  => 문제 %d건' % len(bad))
    txt = '\n'.join(P)
    if outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
            open(os.path.join(outdir, stem + '.txt'), 'w').write(txt + '\n')
        except Exception:
            pass
    return txt, len(bad)
