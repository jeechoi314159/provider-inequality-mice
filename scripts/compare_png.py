"""Compare regenerated figures with a reference folder pixel by pixel.
Usage: python scripts/compare_png.py <reference png folder> [figures/png]"""
import os, sys
import numpy as np
from PIL import Image
ref = sys.argv[1]; new = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(__file__), '..', 'figures', 'png')
bad = 0
for n in sorted(f for f in os.listdir(ref) if f.endswith('.png')):
    p = os.path.join(new, n)
    if not os.path.exists(p):
        print(f'{n:34s} missing'); bad += 1; continue
    a = np.asarray(Image.open(os.path.join(ref, n)).convert('RGB'), dtype=np.int16)
    b = np.asarray(Image.open(p).convert('RGB'), dtype=np.int16)
    if a.shape != b.shape:
        print(f'{n:34s} size differs'); bad += 1; continue
    d = int(np.abs(a - b).max()); bad += d > 2
    print(f'{n:34s} max difference {d}')
print('all identical' if not bad else f'{bad} figure(s) differ')
