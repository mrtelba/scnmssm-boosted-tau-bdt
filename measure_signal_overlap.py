import numpy as np
from sklearn.model_selection import train_test_split

# Replays the random draws of domain_shift.py (same seed, same call
# order) and counts signal events shared between the training set of
# one model and the shifted test set of the other. No BDT is trained.

rng = np.random.default_rng(42)

S    = np.nan_to_num(np.load('evtlevel_sig.npy'))
Z100 = np.nan_to_num(np.load('evtlevel_bkg.npy'))
Z50  = np.nan_to_num(np.load('evtlevel_z50.npy'))

N = 20000

# same rng calls, in the same order, as build() in domain_shift.py
sig100 = rng.choice(len(S),    N, replace=False)
_      = rng.choice(len(Z100), N, replace=False)
sig50  = rng.choice(len(S),    N, replace=False)
_      = rng.choice(len(Z50),  N, replace=False)

y = np.array([1]*N + [0]*N)
rows = np.arange(2*N)

# domain_shift.py splits X100 and X50 with identical y and random_state,
# so the row positions in train and test are the same for both
tr100, te100 = train_test_split(rows, test_size=0.3,
                                random_state=42, stratify=y)
tr50,  te50  = train_test_split(rows, test_size=0.3,
                                random_state=42, stratify=y)
assert np.array_equal(tr100, tr50) and np.array_equal(te100, te50)

def sig_events(sig_idx, part):
    # rows < N are signal; row r holds S[sig_idx[r]]
    return sig_idx[part[part < N]]

train_sig_100 = sig_events(sig100, tr100)
test_sig_100  = sig_events(sig100, te100)
train_sig_50  = sig_events(sig50,  tr50)
test_sig_50   = sig_events(sig50,  te50)

def report(label, train_ev, test_ev):
    shared = np.intersect1d(train_ev, test_ev)
    # also compare by content, in case S contains duplicate rows
    tr_rows = {S[i].tobytes() for i in train_ev}
    by_value = sum(S[i].tobytes() in tr_rows for i in test_ev)
    print(f'{label}')
    print(f'  signal events in training set : {len(train_ev)}')
    print(f'  signal events in shifted test : {len(test_ev)}')
    print(f'  shared (same event index)     : {len(shared)}'
          f'  = {100*len(shared)/len(test_ev):.2f}%')
    print(f'  shared (identical feature row): {by_value}'
          f'  = {100*by_value/len(test_ev):.2f}%')

print(f'signal pool: {len(S)} events, '
      f'expected overlap if independent: '
      f'{100*len(train_sig_100)/len(S):.2f}%\n')
report('m100 (trained on pT>100) tested on pT>50',
       train_sig_100, test_sig_50)
report('m50  (trained on pT>50) tested on pT>100',
       train_sig_50, test_sig_100)

# for reference: overlap within the same slice must be zero
print(f'\nsanity: same-slice train/test overlap = '
      f'{len(np.intersect1d(train_sig_100, test_sig_100))}, '
      f'{len(np.intersect1d(train_sig_50, test_sig_50))}')
