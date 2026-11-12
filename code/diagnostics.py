"""Diagnostics layer: reads the saved experiment data in ../data and writes
../data/summary.json with every number quoted in Section 7.  No simulation and
no plotting happens here (simulation: experiments.py, fig_experiments.py;
plotting: make_figures.py)."""
import json, os, numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data') + '/'
S = {}


def load_time():
    d = json.load(open(D + 'time.json'))
    for f in ['time2.json', 'time_armijo.json']:
        if os.path.exists(D + f):
            d.update(json.load(open(D + f)))
    return d


# ---------------------------------------------------------------- Figure 1
d = load_time()
f1 = {}
for k, v in d.items():
    tau = np.array([r['tau'] for r in v]); e = np.array([r['Hm1max'] for r in v])
    rates = np.log(e[:-1] / e[1:]) / np.log(tau[:-1] / tau[1:])
    f1[k] = dict(tau=tau.tolist(), err=e.tolist(), rates=rates.tolist(),
                 err_ratio_to_exact=(e / np.array([r['Hm1max'] for r in d['exact']])).tolist(),
                 iters=[r['iters'] for r in v])
S['fig1'] = f1

# ---------------------------------------------------------------- Figure 2
R = json.load(open(D + 'fig_resid.json'))
f2 = {}
for key in ['A0', 'A2']:
    blk = R[key]; out = dict(t=blk['t'], theta_N=blk['theta_N'])
    for m, h in blk['methods'].items():
        y = np.array(h['RB']); x = np.array(h['nfft']); pa = h['practical_accept']
        fin = np.isfinite(y)
        tight = np.where(y <= 1e-10)[0]
        out[m] = dict(diverged=bool((~fin).any() or y[fin][-1] > 1),
                      practical_iter=pa, practical_nfft=None if pa is None else int(x[pa]),
                      practical_RB=None if pa is None else float(y[pa]),
                      tight_iter=int(tight[0]) if len(tight) else None,
                      tight_nfft=int(x[tight[0]]) if len(tight) else None,
                      max_RB=float(np.nanmax(y[fin])))
    f2[key] = out
W = json.load(open(D + 'fig_work.json'))
rows = W['rows']; taus = sorted({r['tau'] for r in rows}, reverse=True)
work = {}
for tau in taus:
    rr = {r['method']: r for r in rows if r['tau'] == tau}
    errs = [r['errL2'] for r in rr.values()]
    work[str(tau)] = dict(errL2_range=[min(errs), max(errs)], errL2_spread=max(errs) / min(errs) - 1,
                          nfft={m: r['nfft_total'] for m, r in rr.items()},
                          fails={m: r['fails'] for m, r in rr.items()},
                          ratio_to_newton_tight={m: r['nfft_total'] / rr['newton_tight']['nfft_total'] for m, r in rr.items()},
                          ratio_to_code_K6={m: r['nfft_total'] / rr['code_K6']['nfft_total'] for m, r in rr.items()})
e = [work[str(t)]['errL2_range'][0] for t in taus]
work['observed_order'] = [float(np.log(e[i] / e[i + 1]) / np.log(taus[i] / taus[i + 1])) for i in range(len(taus) - 1)]
work['meta'] = W['meta']
f2['work'] = work
M = json.load(open(D + 'mesh.json'))
f2['mesh'] = [{k: r[k] for k in r if k in ('method', 'N', 'nfft_per_step', 'iters_mean', 'iters_max', 'mean', 'max')} for r in M]
S['fig2'] = f2

# ---------------------------------------------------------------- Figure 3
pts = json.load(open(D + 'sharp.json'))
eff = np.array([p['eB'] / p['RB'] for p in pts]); om = np.array([p['omega'] for p in pts]); q = np.array([p['q'] for p in pts])
A = json.load(open(D + 'fig_accept.json'))
ar = np.array([p['err'] / p['tol'] for p in A['pts']])
S['fig3'] = dict(n_iterates=len(pts), eff_max=float(eff.max()), eff_min=float(eff.min()),
                 eff_min_w1=float(eff[om == 1].min()), eff_min_w05=float(eff[om == 0.5].min()),
                 q_max=float(q.max()), lower=float(1 / (1 + q.max())),
                 n_accept=len(ar), accept_violations=int((ar > 1).sum()), accept_max_err_over_tol=float(ar.max()),
                 accept_min_err_over_tol=float(ar.min()), accept_meta=A['meta'])

# ---------------------------------------------------------------- Figures 4-5
if os.path.exists(D + 'fig_bench.json'):
    B = json.load(open(D + 'fig_bench.json')); r = {k: np.array(v) for k, v in B['rec'].items()}
    t = r['t']; n = len(t)
    dEp = np.diff(np.concatenate([[B['meta'].get('Etil0', r['Etil_p'][0])], r['Etil_p']]))
    dEt = np.diff(np.concatenate([[B['meta'].get('Etil0', r['Etil_t'][0])], r['Etil_t']]))
    f5 = dict(steps=n, T=float(t[-1]),
              max_increment_p=float(np.diff(r['Etil_p']).max()), max_increment_t=float(np.diff(r['Etil_t']).max()),
              n_increase_p=int((np.diff(r['Etil_p']) > 0).sum()), n_increase_t=int((np.diff(r['Etil_t']) > 0).sum()),
              mass_drift_p=float(np.abs(r['mass_p'] - r['mass_p'][0]).max()),
              mass_drift_t=float(np.abs(r['mass_t'] - r['mass_t'][0]).max()),
              rel_energy_diff_max=float(np.max(np.abs(r['Etil_p'] - r['Etil_t']) / np.abs(r['Etil_t'] + B['meta'].get('area', 12.8**2) / 4))),
              diffL2_max=float(r['diffL2'].max()), t_diffL2_max=float(t[r['diffL2'].argmax()]),
              diffL2_end=float(r['diffL2'][-1]), diffInf_max=float(r['diffInf'].max()),
              diffInf_end=float(r['diffInf'][-1]), diffH1_max=float(r['diffH1'].max()), diffH1_end=float(r['diffH1'][-1]),
              it_p_mean=float(r['it_p'].mean()), it_p_max=int(r['it_p'].max()),
              it_t_mean=float(r['it_t'].mean()), it_t_max=int(r['it_t'].max()),
              fft_p_mean=float(r['fft_p'].mean()), fft_t_mean=float(r['fft_t'].mean()),
              fft_ratio=float(r['fft_t'].sum() / r['fft_p'].sum()),
              TE_ratio_max=float(r['TEratio_p'].max()), TE_ratio_median=float(np.median(r['TEratio_p'])),
              accepted_all=bool(r['acc_p'].all()), R2_p_max=float(r['R2_p'].max()), RB_p_max=float(r['RB_p'].max()),
              phi_max=float(r['max_p'].max()), phi_min=float(r['min_p'].min()))
    # coarsening rate of the energy between t=1 and the final time
    Ebar = r['Ep'] + 0 * t
    s = (t >= 1.0)
    if s.sum() > 10:
        Es = r['Etil_p'][s] + B['meta'].get('area', 12.8**2) / 4
        f5['energy_slope_t1_T'] = float(np.polyfit(np.log(t[s]), np.log(Es), 1)[0])
    S['bench'] = f5
    S['bench_meta'] = B['meta']


# ---------------------------------------------------------------- Figures 6-7: interface evolution
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig_experiments2 import interface_disp


def _l2(a, b, L):
    return float(np.sqrt(((a - b)**2).mean()) * L)


def _events(t, c):
    return [float(t[i]) for i in range(1, len(c)) if c[i] != c[i - 1]]


if os.path.exists(D + 'fig_star_refine.json') and os.path.exists(D + 'fig_star_prod.json'):
    SR = json.load(open(D + 'fig_star_refine.json')); Ssn = np.load(D + 'fig_star_refine_snaps.npz'); L = SR['meta']['L']
    ref = 'N128_tau0.00025'
    st = {'reference': ref, 'runs': {}}
    for key, run in SR['runs'].items():
        e = dict(N=run['N'], tau=run['tau'], nfft=run['nfft'], mergers=_events(np.array(run['t']), run['ncomp']))
        for tk in ['0.1', '0.25', '0.5']:
            a = Ssn[f'{key}_{tk}']
            b = Ssn[f'{ref}_{tk}']
            if a.shape != b.shape:                          # compare on the common (coarse) grid points
                a = a[::a.shape[0] // b.shape[0], ::a.shape[0] // b.shape[0]]
            if key != ref:
                e[f'L2_{tk}'] = _l2(a, b, L); e[f'delta_{tk}'] = interface_disp(a, b, L)
        st['runs'][key] = e
    for tk in ['0.1', '0.25', '0.5']:
        a = Ssn[f'N128_tau0.001_{tk}']; b = Ssn[f'N128_tau0.0005_{tk}']
        st[f'tau_vs_half_{tk}'] = dict(L2=_l2(a, b, L), delta=interface_disp(a, b, L))
    P = json.load(open(D + 'fig_star_prod.json')); r = P['rec']; t = np.array(r['t'])
    Sp = np.load(D + 'fig_star_prod_snaps.npz')
    st['prod'] = dict(meta=P['meta'], fft_p=int(sum(r['fft_p'])), fft_t=int(sum(r['fft_t'])),
                      ratio=sum(r['fft_t']) / sum(r['fft_p']), fft_p_mean=float(np.mean(r['fft_p'])), fft_t_mean=float(np.mean(r['fft_t'])),
                      it_p_mean=float(np.mean(r['it_p'])), it_p_max=int(max(r['it_p'])), it_t_mean=float(np.mean(r['it_t'])), it_t_max=int(max(r['it_t'])),
                      diffL2_max=max(r['diffL2']), diffInf_max=max(r['diffInf']), diffH1_max=max(r['diffH1']), delta_at=r['delta_at'],
                      mergers_p=_events(t, r['ncomp_p']), mergers_t=_events(t, r['ncomp_t']),
                      TE_max=max(r['TEratio_p']), acc_all=all(r['acc_p']),
                      Etil_increases=[int((np.diff(r['Etil_p']) > 0).sum()), int((np.diff(r['Etil_t']) > 0).sum())],
                      max_dEtil=[float(np.diff(r['Etil_p']).max()), float(np.diff(r['Etil_t']).max())],
                      mass_drift=[float(np.abs(r['mass_p']).max()), float(np.abs(r['mass_t']).max())],
                      phi_range=[float(min(r['min_p'])), float(max(r['max_p']))],
                      snap_max_diff={k[2:]: float(np.abs(Sp[k].astype(float) - Sp['t' + k[1:]].astype(float)).max())
                                     for k in Sp.files if k.startswith('p_')})
    S['star'] = st

if os.path.exists(D + 'fig_drop_refine.json') and os.path.exists(D + 'fig_drop_prod.json'):
    DR = json.load(open(D + 'fig_drop_refine.json'))
    dr = {'runs': {k: dict(N=v['N'], tau=v['tau'], nfft=v['nfft'], t_dis=v['t_dis'], R_at=v['R_at']) for k, v in DR['runs'].items()}}
    base = DR['runs'].get('N256_tau0.001')
    for k, v in DR['runs'].items():
        dr['runs'][k]['maxdiff_R_vs_N256_tau0.001_at_outputs'] = max(
            abs(a - b) for tk in v['R_at'] for a, b in zip(v['R_at'][tk], base['R_at'][tk]))
    P = json.load(open(D + 'fig_drop_prod.json')); r = P['rec']; t = np.array(r['t'])
    Rp = np.array(r['R_p']); Rt = np.array(r['R_t']); dR = np.abs(Rp - Rt)
    Sp = np.load(D + 'fig_drop_prod_snaps.npz')
    dr['prod'] = dict(meta=P['meta'], fft_p=int(sum(r['fft_p'])), fft_t=int(sum(r['fft_t'])), ratio=sum(r['fft_t']) / sum(r['fft_p']),
                      fft_p_mean=float(np.mean(r['fft_p'])), fft_t_mean=float(np.mean(r['fft_t'])),
                      it_p_mean=float(np.mean(r['it_p'])), it_p_max=int(max(r['it_p'])), it_t_mean=float(np.mean(r['it_t'])), it_t_max=int(max(r['it_t'])),
                      t_dis_p=P['t_dis_p'], t_dis_t=P['t_dis_t'],
                      maxdR=dR.max(0).tolist(), t_maxdR=[float(t[i]) for i in dR.argmax(0)],
                      R_at={f'{tk:g}': [Rp[np.argmin(abs(t - tk))].tolist(), Rt[np.argmin(abs(t - tk))].tolist()] for tk in P['meta']['snap_times']},
                      diffL2_max=max(r['diffL2']), diffInf_max=max(r['diffInf']), diffH1_max=max(r['diffH1']), delta_at=r['delta_at'],
                      TE_max=max(r['TEratio_p']), acc_all=all(r['acc_p']),
                      Etil_increases=[int((np.diff(r['Etil_p']) > 0).sum()), int((np.diff(r['Etil_t']) > 0).sum())],
                      max_dEtil=[float(np.diff(r['Etil_p']).max()), float(np.diff(r['Etil_t']).max())],
                      mass_drift=[float(np.abs(r['mass_p']).max()), float(np.abs(r['mass_t']).max())],
                      phi_range=[float(min(r['min_p'])), float(max(r['max_p']))],
                      snap_max_diff={k[2:]: float(np.abs(Sp[k].astype(float) - Sp['t' + k[1:]].astype(float)).max())
                                     for k in Sp.files if k.startswith('p_')})
    # time-step effect on the radii at the production resolution (tau vs tau/2), at every step of the coarser run
    a, b = DR['runs']['N256_tau0.001'], DR['runs']['N256_tau0.0005']
    ta, Ra, tb, Rb = np.array(a['t']), np.array(a['R']), np.array(b['t']), np.array(b['R'])
    idx = np.searchsorted(tb, ta - 1e-12); both = (Ra > 0) & (Rb[idx] > 0)
    dd = np.where(both, np.abs(Ra - Rb[idx]), 0.0)
    dr['tau_effect_R_max'] = dd.max(0).tolist()
    dr['tau_effect_R_median'] = [float(np.median(dd[both[:, j], j])) for j in range(3)]
    dr['prod_dR_median'] = [float(np.median(dR[(Rp[:, j] > 0) & (Rt[:, j] > 0), j])) for j in range(3)]
    S['drops'] = dr

if os.path.exists(D + 'fig_seeds.json'):
    B = json.load(open(D + 'fig_seeds.json')); Ss = np.load(D + 'fig_seeds_snaps.npz'); L = B['meta']['L']

    def ev(t, c):
        return [(float(t[i]), int(c[i - 1]), int(c[i])) for i in range(1, len(c)) if c[i] != c[i - 1]]

    def match(e1, e2, tau):
        """pair the k-th change in the component count of the two runs; return the time shifts in steps"""
        return [round((a[0] - b[0]) / tau) for a, b in zip(e1, e2)] if len(e1) == len(e2) else None
    sd = {}
    for k, v in B['runs'].items():
        r = v['rec']; t = np.array(r['t'])
        ep, et = ev(t, r['ncomp_p']), ev(t, r['ncomp_t'])
        sh = match(ep, et, B['meta']['tau'])
        cp_, ct_ = np.array(r['ncomp_p']), np.array(r['ncomp_t'])
        sd[k] = dict(fft_p=int(sum(r['fft_p'])), fft_t=int(sum(r['fft_t'])), ratio=sum(r['fft_t']) / sum(r['fft_p']),
                     count_steps_differ=int((cp_ != ct_).sum()), count_max_diff=int(np.abs(cp_ - ct_).max()),
                     ncomp_max=int(max(cp_.max(), ct_.max())), t_ncomp_max=float(t[int(np.argmax(cp_))]),
                     it_t_cap=60,
                     fails=r['acc_p'].count(False), events_p=len(ep), events_t=len(et),
                     events_shifted=None if sh is None else int(sum(x != 0 for x in sh)),
                     max_shift_steps=None if sh is None else int(max(abs(x) for x in sh)),
                     ncomp_at={f'{x:g}': [r['ncomp_p'][int(np.argmin(abs(t - x)))], r['ncomp_t'][int(np.argmin(abs(t - x)))]] for x in B['meta']['snap_times']},
                     delta_at=r['delta_at'], diffL2_max=max(r['diffL2']), t_diffL2_max=float(t[int(np.argmax(r['diffL2']))]),
                     diffL2_median=float(np.median(r['diffL2'])), diffInf_max=max(r['diffInf']),
                     TE_max=max(r['TEratio_p']), Etil_increases=[int((np.diff(r['Etil_p']) > 0).sum()), int((np.diff(r['Etil_t']) > 0).sum())],
                     mass_drift=[float(np.abs(r['mass_p']).max()), float(np.abs(r['mass_t']).max())],
                     it_p_mean=float(np.mean(r['it_p'])), it_p_max=int(max(r['it_p'])), it_t_mean=float(np.mean(r['it_t'])), it_t_max=int(max(r['it_t'])),
                     phi_range=[float(min(r['min_p'])), float(max(r['max_p']))])
    if os.path.exists(D + 'fig_seed_refine.json'):
        RJ = json.load(open(D + 'fig_seed_refine.json')); RF = np.load(D + 'fig_seed_refine_snaps.npz')
        r1 = B['runs']['seed1']['rec']; t1 = np.array(r1['t']); e1 = ev(t1, r1['ncomp_t'])
        ref = {}
        for key, run in RJ['runs'].items():
            tt = np.array(run['t']); e2 = ev(tt, run['ncomp'])
            d = {}
            for x in B['meta']['snap_times']:
                q = RF[f'{key}_{x:g}'].astype(float); q = q[::q.shape[0] // 256, ::q.shape[0] // 256]
                b = Ss[f'seed1_t_{x:g}'].astype(float)
                d[f'{x:g}'] = dict(L2=_l2(q, b, L), delta=interface_disp(q, b, L), ncomp=int(run['ncomp'][int(np.argmin(abs(tt - x)))]))
            dt = [a[0] - b[0] for a, b in zip(e2, e1)] if len(e1) == len(e2) else None
            cr = np.array(run['ncomp']); stride = int(round(B['meta']['tau'] / run['tau']))
            cr = cr[stride - 1::stride]; c1 = np.array(r1['ncomp_t'])
            ref[key] = dict(at=d, events=len(e2), events_ref=len(e1), count_steps_differ=int((cr != c1).sum()),
                            count_max_diff=int(np.abs(cr - c1).max()),
                            max_event_shift=None if dt is None else float(max(abs(x) for x in dt)),
                            median_event_shift=None if dt is None else float(np.median(np.abs(dt))))
        sd['seed1_refine'] = ref
    S['seeds'] = sd

json.dump(S, open(D + 'summary.json', 'w'), indent=1, default=float)


if __name__ == '__main__':
    import pprint
    for k in ['star', 'drops', 'seeds']:
        if k in S and k != 'drops':
            pprint.pprint({a: b for a, b in S[k].items() if a != 'meta'}, width=160, compact=True)
    for k in ['fig3', 'bench']:
        if k in S:
            pprint.pprint({a: b for a, b in S[k].items() if 'meta' not in a})
    for k in ['A0', 'A2']:
        pprint.pprint(S['fig2'][k])
    pprint.pprint({t: (S['fig2']['work'][t]['errL2_spread'], {m: round(v, 2) for m, v in S['fig2']['work'][t]['ratio_to_code_K6'].items()})
                   for t in map(str, taus)})
    print('order', S['fig2']['work']['observed_order'])
    for k in ['exact', 'w05_K1', 'w05_K2', 'w05_K4', 'w05_acc', 'armijo_acc', 'armijo_acc01', 'w1_K1', 'w1_K6']:
        if k in f1:
            print(k, ['%.2f' % x for x in f1[k]['rates']], 'ratio', ['%.2f' % x for x in f1[k]['err_ratio_to_exact']][-3:],
                  'its', f1[k]['iters'][-3:])
    print('mesh', S['fig2']['mesh'][:3])
