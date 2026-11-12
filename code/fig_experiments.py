"""Simulations for the figures (data only; plotting lives in make_figures.py).

Usage: python3 fig_experiments.py <name>    with name in {resid, work, accept, bench}
Each run writes ../data/fig_<name>.json (and .npz for fields) including a 'meta' block that records
grid, domain, eps, A, tau, seed, reference settings and norm conventions.

Norm conventions (discrete, h^2-weighted, see Section 2 of the paper):
  ||f||        discrete L^2 NORM
  ||f||_{-1}   discrete H^{-1} norm of the mean-zero part
  ||f||_{B^-1} = (tau^-1 ||f||_{-1}^2 + A ||f||^2 + 3/4 eps^2 |f|_1^2)^{1/2}, |f|_1 the H^1 SEMINORM
"""
import sys, json, time
import numpy as np
from chsolver import CH

DATA = '../data/'
NORMS = {'L2': 'discrete L2 norm', 'Hm1': 'discrete H^-1 norm (mean-zero part)',
         'Binv': 'B^-1 norm (contains the H^1 seminorm)', 'B': 'dual B norm', 'H1': 'H^1 seminorm'}


def rand_field(N, seed, amp=0.1):
    rng = np.random.default_rng(seed)
    return amp * (2 * rng.random((N, N)) - 1)


def hm1(S, e):
    eh = np.fft.rfft2(e - e.mean())
    return float(np.sqrt(S.ipF(eh, eh, S.invs)))


def binv(S, e):
    eh = np.fft.rfft2(e - e.mean())
    return float(np.sqrt(S.ipF(eh, eh, S.Binv)))


def l2(S, e):
    return float(np.sqrt(S.ip(e, e)))


# ---------------------------------------------------------------- Figure 2(a)
def exp_resid():
    """Residual histories of four inner methods from the same hard step (spinodal stage)."""
    N, L, eps, tau, seed = 256, 6.4, 0.02, 0.004, 1
    out = {'meta': dict(N=N, L=L, eps=eps, tau=tau, seed=seed, norms=NORMS,
                        note='history phi^{n-1}, phi^n produced by Newton with ||R||_B<=1e-12 from seed-1 data; '
                             'phi^{-1}=phi^0. Same (phi^n,phi^{n-1}) for every method.')}
    for A in [0.0, 2.0]:
        S = CH(N, L, eps, A, tau)
        phi = rand_field(N, seed); phim = phi.copy()
        nstar = 20                                   # t_n = 0.08, onset of phase separation
        for n in range(nstar):
            new, _ = S.step(phi, phim, n * tau, method='newton', tolB=1e-12, kmax=60)
            phim, phi = phi, new
        res = {}
        for name, kw in [('linear_w1', dict(method='relax', omega=1.0)), ('relax_w05', dict(method='relax', omega=0.5)),
                         ('armijo', dict(method='armijo', omega=1.0)), ('newton', dict(method='newton'))]:
            S.nfft = 0
            _, info = S.step(phi, phim, nstar * tau, tolB=1e-12, kmax=150, record=True, **kw)
            h = info['hist']
            # where the practical rule (T_E with eta=1/2, T_A with c_R=1) would accept
            acc = next((k for k, r in enumerate(h) if r['R2'] <= tau**2 and r['pert'] <= 0.5 * r['Dn']), None)
            res[name] = dict(RB=[r['RB'] for r in h], nfft=[r['nfft'] for r in h], accepted=info['accepted'],
                             practical_accept=acc, maxabs_hist=float(np.abs(phi).max()))
            print(A, name, len(h), h[-1]['RB'], 'practical at', acc, flush=True)
        out[f'A{A:g}'] = dict(t=nstar * tau, theta_N=S.theta_N, methods=res)
    json.dump(out, open(DATA + 'fig_resid.json', 'w'))


# ---------------------------------------------------------------- Figure 2(b)
def run_to_T(S, phi0, T, sol, keep=False):
    phi, phim = phi0.copy(), phi0.copy()
    ns = int(round(T / S.tau))
    S.nfft = 0
    fails = 0
    for n in range(ns):
        new, info = S.step(phi, phim, n * S.tau, **sol)
        fails += (not info['accepted'])
        if not np.all(np.isfinite(new)) or np.abs(new).max() > 1e3:
            return None, S.nfft, n + 1, fails
        phim, phi = phi, new
    return phi, S.nfft, ns, fails


def shapes_ic(N, L, eps):
    """Smooth interface-relaxation data: a six-lobed star and two circles, tanh profiles of width sqrt(2) eps."""
    x = np.arange(N) * L / N
    X, Y = np.meshgrid(x, x, indexing='ij')
    cx, cy = 0.32 * L, 0.5 * L
    r = np.sqrt((X - cx)**2 + (Y - cy)**2); th = np.arctan2(Y - cy, X - cx)
    star = (0.22 * L + 0.04 * L * np.cos(6 * th)) - r
    c1 = 0.11 * L - np.sqrt((X - 0.70 * L)**2 + (Y - 0.36 * L)**2)
    c2 = 0.11 * L - np.sqrt((X - 0.70 * L)**2 + (Y - 0.64 * L)**2)
    return np.tanh(np.maximum(np.maximum(star, c1), c2) / (np.sqrt(2) * eps))


def exp_work():
    """Error at T against a fine reference versus total transforms, for several solvers and time steps.
    Smooth interface-relaxation problem (star + two merging circles), for which tau-convergence is asymptotic."""
    N, L, eps, A, T = 128, 2.0, 0.04, 2.0, 0.5
    phi0 = shapes_ic(N, L, eps)
    np.save(DATA + 'phi0_work_shapes_128.npy', phi0)
    tref = 6.25e-5
    t0 = time.time()
    Sref = CH(N, L, eps, A, tref)
    ref, _, _, _ = run_to_T(Sref, phi0, T, dict(method='newton', tolB=1e-12, kmax=60))
    S2 = CH(N, L, eps, A, 2 * tref)
    ref2, _, _, _ = run_to_T(S2, phi0, T, dict(method='newton', tolB=1e-12, kmax=60))
    floor = l2(Sref, ref2 - ref) / 3                 # Richardson estimate of the reference error (second order)
    print('reference done', time.time() - t0, 'floor', floor, flush=True)
    np.save(DATA + 'fig_work_ref.npy', ref)
    methods = {'code_K6': dict(method='fixed', omega=1.0, K=6),
               'linear_tight': dict(method='relax', omega=1.0, tolB=1e-10, kmax=300),
               'armijo_pract': dict(method='armijo', omega=1.0, cR=1.0, eta=0.5, kmax=300),
               'armijo_tight': dict(method='armijo', omega=1.0, tolB=1e-10, kmax=300),
               'newton_pract': dict(method='newton', cR=1.0, eta=0.5, kmax=60),
               'newton_tight': dict(method='newton', tolB=1e-10, kmax=60)}
    rows = []
    for tau in [0.008, 0.004, 0.002, 0.001, 0.0005]:
        for name, sol in methods.items():
            S = CH(N, L, eps, A, tau)
            phi, nfft, steps, fails = run_to_T(S, phi0, T, sol)
            r = dict(method=name, tau=tau, nfft_total=nfft, steps=steps, fails=fails, diverged=phi is None)
            if phi is not None:
                e = phi - ref
                r.update(errL2=l2(S, e), errHm1=hm1(S, e), errInf=float(np.abs(e).max()))
            rows.append(r); print(r, flush=True)
            json.dump(dict(meta=dict(problem='star (6 lobes) + two merging circles, tanh profiles', N=N, L=L, eps=eps, A=A, T=T,
                                     tau_ref=tref, ref_solver='Newton, ||R||_B<=1e-12', ref_error_L2_estimate=floor,
                                     error='discrete L2 NORM of phi(T)-phi_ref(T)', ref_norm_L2=l2(Sref, ref), norms=NORMS),
                           rows=rows), open(DATA + 'fig_work.json', 'w'), indent=1)


# ---------------------------------------------------------------- Figure 3(b)
def exp_accept():
    """True inner-solve error of the iterate accepted by ||R||_B <= tol, for several tolerances and methods."""
    N, L, eps, A, tau, seed = 128, 3.2, 0.02, 2.0, 0.004, 3
    S = CH(N, L, eps, A, tau)
    phi = rand_field(N, seed); phim = phi.copy()
    pts = []
    tols = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    for n in range(250):
        ref, _ = S.step(phi, phim, 0, method='newton', tolB=1e-13, kmax=60)
        if n % 10 == 5:
            for name, kw in [('linear_w1', dict(method='relax', omega=1.0)), ('relax_w05', dict(method='relax', omega=0.5)),
                             ('armijo', dict(method='armijo', omega=1.0)), ('newton', dict(method='newton'))]:
                for tol in tols:
                    v, info = S.step(phi, phim, 0, tolB=tol, kmax=300, record=True, **kw)
                    if info['accepted']:
                        pts.append(dict(n=n, method=name, tol=tol, RB=info['hist'][-1]['RB'], err=binv(S, v - ref),
                                        iters=info['iters']))
        phim, phi = phi, ref
    viol = sum(p['err'] > p['RB'] * (1 + 1e-8) for p in pts)
    print('points', len(pts), 'violations err > RB:', viol, 'max err/tol', max(p['err'] / p['tol'] for p in pts))
    json.dump(dict(meta=dict(N=N, L=L, eps=eps, A=A, tau=tau, seed=seed, steps='every 10th of 250, starting at 5',
                             reference='Newton with ||R||_B<=1e-13', error='B^-1 norm', norms=NORMS), pts=pts),
              open(DATA + 'fig_accept.json', 'w'))


# ---------------------------------------------------------------- Figures 4 and 5
def exp_bench():
    """Lockstep run of the practical solver and a tight reference solver on the 512^2 benchmark, recording
    per-step energies, mass drift, discrepancy, work, accepted residuals and snapshots."""
    N, L, eps, A, tau, T, seed = 512, 12.8, 0.02, 0.0, 0.004, 10.0, 1
    phi0 = np.load(DATA + 'phi0_seed1_512.npy')
    Sp = CH(N, L, eps, A, tau); St = CH(N, L, eps, A, tau)
    pract = dict(method='armijo', omega=1.0, cR=1.0, eta=0.5, kmax=300, record=True)
    tight = dict(method='newton', tolB=1e-10, kmax=60, record=True)
    pp, ppm = phi0.copy(), phi0.copy(); tt, ttm = phi0.copy(), phi0.copy()
    m0 = float(phi0.mean())
    rec = {k: [] for k in ['t', 'Ep', 'Et', 'Etil_p', 'Etil_t', 'mass_p', 'mass_t', 'diffL2', 'diffInf', 'diffH1',
                           'it_p', 'it_t', 'fft_p', 'fft_t', 'R2_p', 'RB_p', 'TEratio_p', 'max_p', 'min_p', 'acc_p']}
    snaps = {}
    stimes = [0.2, 1.0, 4.0, 10.0]
    ns = int(round(T / tau))
    t0 = time.time()
    for n in range(ns):
        f0 = Sp.nfft
        new_p, ip = Sp.step(pp, ppm, n * tau, **pract)
        fp = Sp.nfft - f0
        f0 = St.nfft
        new_t, it_ = St.step(tt, ttm, n * tau, **tight)
        ft = St.nfft - f0
        ppm, pp = pp, new_p; ttm, tt = tt, new_t
        d = pp - tt
        dh = np.fft.rfft2(d)
        last = ip['hist'][-1]
        rec['t'].append((n + 1) * tau)
        rec['Ep'].append(Sp.energy(pp)); rec['Et'].append(St.energy(tt))
        rec['Etil_p'].append(Sp.mod_energy(pp, ppm)); rec['Etil_t'].append(St.mod_energy(tt, ttm))
        rec['mass_p'].append(float(pp.mean()) - m0); rec['mass_t'].append(float(tt.mean()) - m0)
        rec['diffL2'].append(l2(Sp, d)); rec['diffInf'].append(float(np.abs(d).max()))
        rec['diffH1'].append(float(np.sqrt(Sp.ipF(dh, dh, Sp.s))))
        rec['it_p'].append(ip['iters']); rec['it_t'].append(it_['iters'])
        rec['fft_p'].append(fp); rec['fft_t'].append(ft)
        rec['R2_p'].append(last['R2']); rec['RB_p'].append(last['RB'])
        rec['TEratio_p'].append(last['pert'] / last['Dn'] if last['Dn'] > 0 else 0.0)
        rec['max_p'].append(float(pp.max())); rec['min_p'].append(float(pp.min())); rec['acc_p'].append(bool(ip['accepted']))
        for st in stimes:
            if abs((n + 1) * tau - st) < 0.5 * tau:
                snaps[f'p_{st:g}'] = pp.astype(np.float32); snaps[f't_{st:g}'] = tt.astype(np.float32)
        if (n + 1) % 250 == 0:
            print(n + 1, time.time() - t0, rec['diffL2'][-1], flush=True)
            json.dump(dict(meta=meta_bench(N, L, eps, A, tau, T, seed, m0), rec=rec), open(DATA + 'fig_bench.json', 'w'))
    np.savez_compressed(DATA + 'fig_bench_snaps.npz', **snaps)
    json.dump(dict(meta=meta_bench(N, L, eps, A, tau, T, seed, m0), rec=rec), open(DATA + 'fig_bench.json', 'w'))


def meta_bench(N, L, eps, A, tau, T, seed, m0):
    return dict(N=N, L=L, eps=eps, A=A, tau=tau, T=T, seed=seed, initial='uniform in [-0.1,0.1], ../data/phi0_seed1_512.npy',
                mean0=m0, practical='Armijo (c=1/4), tests (T_E) eta=1/2 and (T_A) c_R=1',
                tight='Newton-PCG (CG rel. tol 1e-2), ||R||_B<=1e-10', diffL2='discrete L2 NORM', diffH1='H^1 SEMINORM',
                norms=NORMS)


def exp_tauscale():
    """Size of the time-discretisation effect on the benchmark: tight solves with tau/2 = 0.002 from the
    same data, snapshots at t = 0.2 and 1 (compared with the tau = 0.004 tight snapshots of exp_bench)."""
    N, L, eps, A, tau, seed = 512, 12.8, 0.02, 0.0, 0.002, 1
    phi0 = np.load(DATA + 'phi0_seed1_512.npy')
    S = CH(N, L, eps, A, tau); p, pm = phi0.copy(), phi0.copy(); snaps = {}
    for n in range(500):
        new, _ = S.step(p, pm, n * tau, method='newton', tolB=1e-10, kmax=60); pm, p = p, new
        for st in [0.2, 1.0]:
            if abs((n + 1) * tau - st) < 0.5 * tau:
                snaps[f'h_{st:g}'] = p.copy()
    np.savez_compressed(DATA + 'fig_tauscale.npz', **snaps)
    json.dump(dict(N=N, L=L, eps=eps, A=A, tau=tau, seed=seed, solver='Newton-PCG, ||R||_B<=1e-10',
                   note='compare with t_0.2, t_1 in fig_bench_snaps.npz (tau=0.004, same solver)'),
              open(DATA + 'fig_tauscale.json', 'w'))


if __name__ == '__main__':
    globals()['exp_' + sys.argv[1]]()
