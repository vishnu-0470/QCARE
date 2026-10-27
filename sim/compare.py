import sys
g = [l.split() for l in open(sys.argv[1])]
r = [l.split() for l in open(sys.argv[2])]
print(f"golden beats={len(g)}  rtl beats={len(r)}")
bad = 0
n = min(len(g), len(r))
for i in range(n):
    if g[i] != r[i]:
        bad += 1
        if bad <= 5:
            print(f"MISMATCH beat {i}:\n  golden: {' '.join(g[i])}\n  rtl   : {' '.join(r[i])}")
print("BIT-EXACT" if bad == 0 and len(g) == len(r) else f"FAIL: {bad} mismatching beats, count diff {len(g)-len(r)}")
sys.exit(0 if bad == 0 and len(g) == len(r) else 1)
