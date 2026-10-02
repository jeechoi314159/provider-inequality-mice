# -*- coding: utf-8 -*-
"""보충 그림 최종 번호 부여 (2026-10-01 v2): 원본 파일은 그대로 두고 figS10–S15 사본을 만든다.
figS07 build_figS07.py, figS08–S09 build_lfp_summary.py, figS16 build_r67.py 가 직접 저장."""
import os, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); SUB = os.path.dirname(HERE)
MAP = {'EvI_entry_auc': 'figS11', 'figS_decoder_variants': 'figS12',
       'figS_tilt_replication': 'figS14', 'figS_dreadd_pooled': 'figS15',
       'figS_dreadd_manipulation_all': 'figS16', 'cand_dreadd_C_timeline': 'figS17'}
# 2026-10-01 v3: figS08·figS09 build_lfp_summary.py, figS10 build_follower_spectro.py (추종자 사건 정렬, 탐색적),
# figS13 build_r67.py (옛 Fig. 4 두 신호 분석). 첫 인용 순서: S1→S17.
for src, dst in MAP.items():
    for ext in ('png', 'pdf'):
        a = os.path.join(SUB, ext, f'{src}.{ext}'); b = os.path.join(SUB, ext, f'{dst}.{ext}')
        if os.path.exists(a):
            shutil.copyfile(a, b); print(f'{dst}.{ext} <- {src}')
        else:
            print('MISSING', a)
