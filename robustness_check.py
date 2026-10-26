import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

rng = np.random.default_rng(42)

JET_NAMES = ['pT','mass','SDmass','tau1','tau21',
             'SubZ','SubAsym','NSubSD','NSubTrim']
EVT_NAMES = ['n_fatjets','HT','MET','n_tau']
for k in range(1,4):
    EVT_NAMES += [f'j{k}_{v}' for v in JET_NAMES]
EVT_NAMES += ['dR_j1j2']

Es = np.load('evtlevel_sig.npy')
Eb = np.load('evtlevel_bkg.npy')

n = min(len(Es), len(Eb))
Es = Es[rng.choice(len(Es), n, replace=False)]
Eb = Eb[rng.choice(len(Eb), n, replace=False)]

def run(cols, label):
    X = np.vstack([Es[:,cols], Eb[:,cols]])
    y = np.array([1]*n + [0]*n)
    X = np.nan_to_num(X, nan=0.0,
                      posinf=0.0, neginf=0.0)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3,
        random_state=42, stratify=y)
    bdt = GradientBoostingClassifier(
        n_estimators=200, max_depth=3,
        learning_rate=0.1, subsample=0.8,
        random_state=42)
    bdt.fit(Xtr, ytr)
    sc = bdt.predict_proba(Xte)[:,1]
    fpr, tpr, _ = roc_curve(yte, sc)
    a = auc(fpr, tpr)
    print(f'\n=== {label} ===')
    print(f'  n_features = {len(cols)}')
    print(f'  AUC = {a:.4f}')
    for be in [0.10, 0.05]:
        i = np.argmin(np.abs(fpr-be))
        print(f'  sig eff at {be*100:>2.0f}% bkg: {tpr[i]:.3f}')
    imp = sorted(zip(bdt.feature_importances_,
                     [EVT_NAMES[c] for c in cols]),
                 reverse=True)
    print('  top:')
    for v, nm in imp[:6]:
        if v > 0.01:
            print(f'    {nm:<14} {v:.3f}')
    return a

all_cols = list(range(len(EVT_NAMES)))

# A: everything
run(all_cols, 'A: all 32 features')

# B: drop every pT-like and HT variable
drop_B = {'HT'}
for k in range(1,4):
    drop_B.add(f'j{k}_pT')
cols_B = [i for i,nm in enumerate(EVT_NAMES)
          if nm not in drop_B]
run(cols_B, 'B: no pT, no HT')

# C: multiplicity + mass only
keep_C = {'n_fatjets','n_tau','MET','dR_j1j2',
          'j1_SDmass','j2_SDmass','j3_SDmass'}
cols_C = [i for i,nm in enumerate(EVT_NAMES)
          if nm in keep_C]
run(cols_C, 'C: multiplicity + SDmass only')

# D: counting only
keep_D = {'n_fatjets','n_tau'}
cols_D = [i for i,nm in enumerate(EVT_NAMES)
          if nm in keep_D]
run(cols_D, 'D: counting only')
