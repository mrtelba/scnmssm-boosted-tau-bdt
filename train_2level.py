import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

def balance(A, B):
    n = min(len(A), len(B))
    ia = rng.choice(len(A), n, replace=False)
    ib = rng.choice(len(B), n, replace=False)
    return A[ia], B[ib], n

def run(Xs, Xb, names, label):
    Xs, Xb, n = balance(Xs, Xb)
    X = np.vstack([Xs, Xb])
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

    print(f'\n===== {label} =====')
    print(f'balanced sample: {n} + {n}')
    print(f'AUC = {a:.4f}')
    for be in [0.10, 0.05, 0.01]:
        i = np.argmin(np.abs(fpr-be))
        print(f'  sig eff at {be*100:>2.0f}% bkg: {tpr[i]:.3f}')

    imp = sorted(zip(bdt.feature_importances_, names),
                 reverse=True)
    print('  top features:')
    for v, nm in imp[:8]:
        if v > 0.005:
            print(f'    {nm:<14} {v:.3f}')
    return fpr, tpr, a

Js = np.load('jetlevel_sig.npy')
Jb = np.load('jetlevel_bkg.npy')
Es = np.load('evtlevel_sig.npy')
Eb = np.load('evtlevel_bkg.npy')

f1, t1, a1 = run(Js, Jb, JET_NAMES,
                 'LEVEL 1 : single fat jet')
f2, t2, a2 = run(Es, Eb, EVT_NAMES,
                 'LEVEL 2 : full event')

fig, ax = plt.subplots(figsize=(7,5.5))
ax.plot(f1, t1, 'g-',  lw=2,
        label=f'Jet-level (AUC={a1:.3f})')
ax.plot(f2, t2, 'b-',  lw=2,
        label=f'Event-level (AUC={a2:.3f})')
ax.plot([0,1],[0,1],'k:',lw=1,label='Random')
ax.set_xlabel('Background Efficiency', fontsize=13)
ax.set_ylabel('Signal Efficiency', fontsize=13)
ax.set_title(r'Boosted $a_1$ discrimination vs $Z\to\tau\tau$',
             fontsize=13)
ax.legend(fontsize=10, loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('ROC_2level.png', dpi=150)
print('\nSaved: ROC_2level.png')
