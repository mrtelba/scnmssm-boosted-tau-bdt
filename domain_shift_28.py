import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Cross-slice test of domain_shift.py, repeated with the 28-input set
# of robustness_check.py (no HT, no j{k}_pT). Same seed, same draws and
# same splits as domain_shift.py; only the columns differ. The 32-input
# run is repeated on the same draws as a reference.

rng = np.random.default_rng(42)

JET = ['pT','mass','SDmass','tau1','tau21',
       'SubZ','SubAsym','NSubSD','NSubTrim']
EVT = ['n_fatjets','HT','MET','n_tau']
for k in range(1,4):
    EVT += [f'j{k}_{v}' for v in JET]
EVT += ['dR_j1j2']

S    = np.nan_to_num(np.load('evtlevel_sig.npy'))
Z100 = np.nan_to_num(np.load('evtlevel_bkg.npy'))
Z50  = np.nan_to_num(np.load('evtlevel_z50.npy'))

def build(sig, bkg, n):
    a = sig[rng.choice(len(sig), n, replace=False)]
    b = bkg[rng.choice(len(bkg), n, replace=False)]
    X = np.nan_to_num(np.vstack([a, b]))
    y = np.array([1]*n + [0]*n)
    return X, y

def fit(X, y):
    m = GradientBoostingClassifier(
        n_estimators=200, max_depth=3,
        learning_rate=0.1, subsample=0.8,
        random_state=42)
    m.fit(X, y)
    return m

def evaluate(m, X, y):
    sc = m.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, sc)
    i10 = np.argmin(np.abs(fpr-0.10))
    return auc(fpr, tpr), tpr[i10]

N = 20000

X100, y100 = build(S, Z100, N)
X50,  y50  = build(S, Z50,  N)
Xtr100, Xte100, ytr100, yte100 = train_test_split(
    X100, y100, test_size=0.3, random_state=42, stratify=y100)
Xtr50, Xte50, ytr50, yte50 = train_test_split(
    X50, y50, test_size=0.3, random_state=42, stratify=y50)

drop = {'HT'} | {f'j{k}_pT' for k in range(1,4)}
sets = {
    '32 inputs': list(range(len(EVT))),
    '28 inputs': [i for i,nm in enumerate(EVT) if nm not in drop],
}

res = {}
for name, cols in sets.items():
    m100 = fit(Xtr100[:,cols], ytr100)
    m50  = fit(Xtr50[:,cols],  ytr50)
    res[name] = {
        '100 -> 100': evaluate(m100, Xte100[:,cols], yte100),
        '100 ->  50': evaluate(m100, Xte50[:,cols],  yte50),
        ' 50 ->  50': evaluate(m50,  Xte50[:,cols],  yte50),
        ' 50 -> 100': evaluate(m50,  Xte100[:,cols], yte100),
    }
    print(f'{name}: {len(cols)} columns'
          + ('' if len(cols) == 32 else
             f', dropped {sorted(drop)}'))

print(f'\n{"train -> test":<14}'
      + ''.join(f'{nm:>22}' for nm in sets))
for cell in res['32 inputs']:
    row = f'{cell:<14}'
    for nm in sets:
        a, e = res[nm][cell]
        row += f'{a:>12.4f} ({e:.3f})  '
    print(row)

print('\nchange (shifted - same slice):')
for nm in sets:
    r = res[nm]
    print(f'  {nm}: 100->50 {r["100 ->  50"][0]-r["100 -> 100"][0]:+.4f}'
          f'   50->100 {r[" 50 -> 100"][0]-r[" 50 ->  50"][0]:+.4f}')
