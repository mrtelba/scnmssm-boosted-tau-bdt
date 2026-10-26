import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

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

print('=== sample means ===')
print(f'{"variable":<14}{"signal":>10}'
      f'{"Z pT>100":>10}{"Z pT>50":>10}')
for i, nm in enumerate(EVT):
    if nm in ['n_fatjets','n_tau','HT','MET',
              'j1_pT','j1_SDmass','j2_SDmass',
              'j3_SDmass','dR_j1j2']:
        print(f'{nm:<14}{S[:,i].mean():>10.2f}'
              f'{Z100[:,i].mean():>10.2f}'
              f'{Z50[:,i].mean():>10.2f}')

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

def evaluate(m, X, y, label):
    sc = m.predict_proba(X)[:,1]
    fpr, tpr, _ = roc_curve(y, sc)
    a = auc(fpr, tpr)
    i10 = np.argmin(np.abs(fpr-0.10))
    print(f'  {label:<34} AUC = {a:.4f}'
          f'   eff@10% = {tpr[i10]:.3f}')
    return a

N = 20000

# train on pT>100, test on both
X100, y100 = build(S, Z100, N)
Xtr, Xte, ytr, yte = train_test_split(
    X100, y100, test_size=0.3,
    random_state=42, stratify=y100)
m100 = fit(Xtr, ytr)

X50, y50 = build(S, Z50, N)
_, Xte50, _, yte50 = train_test_split(
    X50, y50, test_size=0.3,
    random_state=42, stratify=y50)

print('\n=== trained on Z with pT(tautau) > 100 GeV ===')
a_same  = evaluate(m100, Xte,   yte,   'tested on pT>100 (same slice)')
a_cross = evaluate(m100, Xte50, yte50, 'tested on pT>50  (shifted)')

# reverse direction
m50 = fit(*train_test_split(
    X50, y50, test_size=0.3,
    random_state=42, stratify=y50)[::2][0:1][0]
    if False else (train_test_split(
    X50, y50, test_size=0.3,
    random_state=42, stratify=y50)[0],
    train_test_split(
    X50, y50, test_size=0.3,
    random_state=42, stratify=y50)[2]))

print('\n=== trained on Z with pT(tautau) > 50 GeV ===')
b_same  = evaluate(m50, Xte50, yte50, 'tested on pT>50  (same slice)')
b_cross = evaluate(m50, Xte,   yte,   'tested on pT>100 (shifted)')

print('\n=== summary ===')
print(f'  train 100 -> test 100 : {a_same:.4f}')
print(f'  train 100 -> test  50 : {a_cross:.4f}'
      f'   (change {a_cross-a_same:+.4f})')
print(f'  train  50 -> test  50 : {b_same:.4f}')
print(f'  train  50 -> test 100 : {b_cross:.4f}'
      f'   (change {b_cross-b_same:+.4f})')
