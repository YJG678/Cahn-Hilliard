"""Simulations for the interface-evolution figures (data only; plotting lives in make_figures.py).

Usage: python3 fig_experiments2.py <name>
  star_refine   tight solves of the star + two circles problem at tau, tau/2, tau/4, tau/8 (N=128) and N=256
  star_prod     practical vs tight in lockstep at the production resolution, snapshots and histories
  drop_pilot    unequal droplets, tight solve, to choose the horizon and output times
  drop_refine   unequal droplets, tight solves at several tau and N
  drop_prod     unequal droplets, practical vs tight in lockstep
  seeds         supplementary: random data with nonzero mean, three seeds (band-limited, stored coefficients)

Geometric observables (all computed from grid values, no plotting):
  ncomp       number of connected components of {phi > 0} on the periodic grid (4-connectivity)
  droplet R_j equivalent radius (A_j/pi)^{1/2}, A_j the area enclosed by the closed zero contour of the
              piecewise-linear (marching-squares) interpolant that is nearest to droplet j's previous centroid;
              droplet j has disappeared at the first step with no such contour
  delta(a,b)  phase-field-based interface-displacement proxy ||a-b||_{L^1} / (2 |Gamma_b|), |Gamma_b| the
              length of the zero contour of b; for equal tanh profiles whose interfaces differ by a small normal
              distance d it equals mean |d| to leading order (bulk and profile differences also contribute)
"""
import sys, json, time
import numpy as np
import contourpy
from scipy import ndimage
from chsolver import CH

DATA = '../data/'
NORMS = {'L2': 'discrete L2 norm', 'Linf': 'maximum norm', 'H1': 'H^1 SEMINORM',
         'delta': 'interface-displacement proxy ||a-b||_L1/(2|Gamma_b|)'}


# ----------------------------------------------------------------------------- initial data
def star_ic(N, L, eps):
    """Six-lobed star and two circles (same as fig_experiments.shapes_ic), tanh profiles of width sqrt(2) eps."""
    x = np.arange(N) * L / N
    X, Y = np.meshgrid(x, x, indexing='ij')
    cx, cy = 0.32 * L, 0.5 * L
    r = np.sqrt((X - cx)**2 + (Y - cy)**2); th = np.arctan2(Y - cy, X - cx)
    star = (0.22 * L + 0.04 * L * np.cos(6 * th)) - r
    c1 = 0.11 * L - np.sqrt((X - 0.70 * L)**2 + (Y - 0.36 * L)**2)
    c2 = 0.11 * L - np.sqrt((X - 0.70 * L)**2 + (Y - 0.64 * L)**2)
    return np.tanh(np.maximum(np.maximum(star, c1), c2) / (np.sqrt(2) * eps))


DROPS = [((1.00, 1.50), 0.16), ((1.45, 0.72), 0.24), ((0.58, 0.62), 0.32)]   # (centre, radius) on [0,2)^2


def drops_ic(N, L, eps, drops=DROPS):
    """Three separated droplets (phi = +1 inside) in phi = -1, signed distance with periodic images, tanh profile."""
    x = np.arange(N) * L / N
    X, Y = np.meshgrid(x, x, indexing='ij')
    d = -np.inf * np.ones_like(X)
    for (cx, cy), R in drops:
        dx = (X - cx + L / 2) % L - L / 2; dy = (Y - cy + L / 2) % L - L / 2
        d = np.maximum(d, R - np.sqrt(dx**2 + dy**2))
    return np.tanh(d / (np.sqrt(2) * eps))


# ----------------------------------------------------------------------------- geometry
def ncomp(phi):
    """Connected components of {phi>0} with periodic identification (4-connectivity)."""
    lab, n = ndimage.label(phi > 0)
    if n == 0:
        return 0
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in list(zip(lab[0, :], lab[-1, :])) + list(zip(lab[:, 0], lab[:, -1])):
        if a and b:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    return len({find(k) for k in range(1, n + 1)})


def zero_contours(phi, L):
    N = phi.shape[0]
    x = np.arange(N) * L / N
    gen = contourpy.contour_generator(x, x, phi.T)
    return [c for c in gen.lines(0.0) if len(c) > 2]


def poly_area_centroid(c):
    x, y = c[:, 0], c[:, 1]
    cr = x[:-1] * y[1:] - x[1:] * y[:-1]
    A = 0.5 * cr.sum()
    if abs(A) < 1e-14:
        return 0.0, c.mean(0)
    cx = ((x[:-1] + x[1:]) * cr).sum() / (6 * A); cy = ((y[:-1] + y[1:]) * cr).sum() / (6 * A)
    return abs(A), np.array([cx, cy])


def contour_length(phi, L):
    return float(sum(np.sqrt((np.diff(c, axis=0)**2).sum(1)).sum() for c in zero_contours(phi, L)))


def interface_disp(a, b, L):
    h = L / a.shape[0]
    return float(np.abs(a - b).sum() * h * h / (2 * contour_length(b, L)))


class DropletTracker:
    """Tracks the zero-contour area of each droplet; see module docstring for the convention."""

    def __init__(self, drops, L, maxjump=0.15):
        self.cent = [np.array(c, float) for c, _ in drops]
        self.alive = [True] * len(drops); self.t_dis = [None] * len(drops); self.L = L; self.maxjump = maxjump

    def __call__(self, phi, t):
        polys = []
        for c in zero_contours(phi, self.L):
            closed = np.allclose(c[0], c[-1])
            if closed:
                polys.append(poly_area_centroid(c))
        R = []
        for j in range(len(self.cent)):
            if not self.alive[j]:
                R.append(0.0); continue
            best = None
            for A, cen in polys:
                dist = np.linalg.norm(cen - self.cent[j])
                if dist < self.maxjump and (best is None or dist < best[0]):
                    best = (dist, A, cen)
            if best is None:
                self.alive[j] = False; self.t_dis[j] = t; R.append(0.0)
            else:
                self.cent[j] = best[2]; R.append(float(np.sqrt(best[1] / np.pi)))
        return R


# ----------------------------------------------------------------------------- runners
def l2(S, e):
    return float(np.sqrt(S.ip(e, e)))


def h1semi(S, e):
    eh = np.fft.rfft2(e)
    return float(np.sqrt(S.ipF(eh, eh, S.s)))


def run_single(N, L, eps, A, tau, T, phi0, solver, snap_times, track=None):
    """One run; returns snapshots at snap_times, per-step histories, total transforms and failures."""
    S = CH(N, L, eps, A, tau)
    p, pm = phi0.copy(), phi0.copy()
    snaps = {'0': phi0.copy()}; hist = {k: [] for k in ['t', 'ncomp', 'Etil', 'mass', 'it', 'fft', 'R']}
    fails = 0; m0 = float(phi0.mean())
    for n in range(int(round(T / tau))):
        f0 = S.nfft
        new, info = S.step(p, pm, n * tau, **solver)
        if not info['accepted']:
            fails += 1
        if not np.all(np.isfinite(new)) or np.abs(new).max() > 1e3:
            return None
        pm, p = p, new
        t = (n + 1) * tau
        hist['t'].append(t); hist['ncomp'].append(ncomp(p)); hist['Etil'].append(S.mod_energy(p, pm))
        hist['mass'].append(float(p.mean()) - m0); hist['it'].append(info['iters']); hist['fft'].append(S.nfft - f0)
        if track is not None:
            hist['R'].append(track(p, t))
        for st in snap_times:
            if abs(t - st) < 0.5 * tau:
                snaps[f'{st:g}'] = p.copy()
    out = dict(snaps=snaps, hist=hist, nfft=S.nfft, fails=fails)
    if track is not None:
        out['t_dis'] = track.t_dis
    return out


PRACT = dict(method='armijo', omega=1.0, cR=1.0, eta=0.5, kmax=300, record=True)
TIGHT = dict(method='newton', tolB=1e-10, kmax=60, record=True)


def run_lockstep(N, L, eps, A, tau, T, phi0, snap_times, drops=None):
    """Practical and tight solvers from identical data; per-step diagnostics as in fig_experiments.exp_bench."""
    Sp = CH(N, L, eps, A, tau); St = CH(N, L, eps, A, tau)
    pp, ppm, tt, ttm = phi0.copy(), phi0.copy(), phi0.copy(), phi0.copy()
    m0 = float(phi0.mean())
    keys = ['t', 'Etil_p', 'Etil_t', 'mass_p', 'mass_t', 'diffL2', 'diffInf', 'diffH1', 'delta', 'ncomp_p', 'ncomp_t',
            'it_p', 'it_t', 'fft_p', 'fft_t', 'TEratio_p', 'R2_p', 'acc_p', 'max_p', 'min_p', 'R_p', 'R_t']
    rec = {k: [] for k in keys}
    trp = DropletTracker(drops, L) if drops else None; trt = DropletTracker(drops, L) if drops else None
    snaps = {'p_0': phi0.copy(), 't_0': phi0.copy()}
    for n in range(int(round(T / tau))):
        f0 = Sp.nfft; new_p, ip = Sp.step(pp, ppm, n * tau, **PRACT); fp = Sp.nfft - f0
        f0 = St.nfft; new_t, it_ = St.step(tt, ttm, n * tau, **TIGHT); ft = St.nfft - f0
        ppm, pp = pp, new_p; ttm, tt = tt, new_t
        t = (n + 1) * tau; d = pp - tt; last = ip['hist'][-1]
        rec['t'].append(t)
        rec['Etil_p'].append(Sp.mod_energy(pp, ppm)); rec['Etil_t'].append(St.mod_energy(tt, ttm))
        rec['mass_p'].append(float(pp.mean()) - m0); rec['mass_t'].append(float(tt.mean()) - m0)
        rec['diffL2'].append(l2(Sp, d)); rec['diffInf'].append(float(np.abs(d).max())); rec['diffH1'].append(h1semi(Sp, d))
        rec['ncomp_p'].append(ncomp(pp)); rec['ncomp_t'].append(ncomp(tt))
        rec['it_p'].append(ip['iters']); rec['it_t'].append(it_['iters']); rec['fft_p'].append(fp); rec['fft_t'].append(ft)
        rec['TEratio_p'].append(last['pert'] / last['Dn'] if last['Dn'] > 0 else 0.0); rec['R2_p'].append(last['R2'])
        rec['acc_p'].append(bool(ip['accepted'])); rec['max_p'].append(float(pp.max())); rec['min_p'].append(float(pp.min()))
        if drops:
            rec['R_p'].append(trp(pp, t)); rec['R_t'].append(trt(tt, t))
        for st in snap_times:
            if abs(t - st) < 0.5 * tau:
                snaps[f'p_{st:g}'] = pp.copy(); snaps[f't_{st:g}'] = tt.copy()
                rec.setdefault('delta_at', {})[f'{st:g}'] = interface_disp(pp, tt, L)
    out = dict(rec=rec)
    if drops:
        out['t_dis_p'] = trp.t_dis; out['t_dis_t'] = trt.t_dis
    return out, snaps


def save(name, meta, data, snaps=None):
    json.dump(dict(meta=meta, **data), open(DATA + f'fig_{name}.json', 'w'), default=float)
    if snaps is not None:
        np.savez_compressed(DATA + f'fig_{name}_snaps.npz', **snaps)


# ----------------------------------------------------------------------------- star problem
STAR = dict(L=2.0, eps=0.04, A=2.0, T=0.5)
STAR_TIMES = [0.1, 0.25, 0.5]


def exp_star_refine():
    L, eps, A, T = STAR['L'], STAR['eps'], STAR['A'], STAR['T']
    runs = [(128, 0.002), (128, 0.001), (128, 0.0005), (128, 0.00025), (256, 0.00025), (256, 0.001)]
    out, snaps = {}, {}
    for N, tau in runs:
        t0 = time.time()
        r = run_single(N, L, eps, A, tau, T, star_ic(N, L, eps), dict(method='newton', tolB=1e-10, kmax=60), STAR_TIMES)
        key = f'N{N}_tau{tau:g}'
        out[key] = dict(N=N, tau=tau, nfft=r['nfft'], fails=r['fails'], ncomp=r['hist']['ncomp'], t=r['hist']['t'],
                        Etil=r['hist']['Etil'])
        for k, v in r['snaps'].items():
            snaps[f'{key}_{k}'] = v
        print(key, 'done', time.time() - t0, 'ncomp changes',
              [(r['hist']['t'][i], r['hist']['ncomp'][i]) for i in range(1, len(r['hist']['t']))
               if r['hist']['ncomp'][i] != r['hist']['ncomp'][i - 1]], flush=True)
        save('star_refine', dict(problem='star (6 lobes) + two circles', L=L, eps=eps, A=A, T=T,
                                 solver='Newton-PCG, ||R||_B<=1e-10', snap_times=STAR_TIMES, norms=NORMS), dict(runs=out), snaps)


def exp_star_prod():
    N, tau = int(sys.argv[2]), float(sys.argv[3])
    L, eps, A, T = STAR['L'], STAR['eps'], STAR['A'], STAR['T']
    times = sorted(set(STAR_TIMES + [0.02, 0.025, 0.03, 0.18, 0.19, 0.2]))
    out, snaps = run_lockstep(N, L, eps, A, tau, T, star_ic(N, L, eps), times)
    save('star_prod', dict(problem='star (6 lobes) + two circles', N=N, L=L, eps=eps, A=A, T=T, tau=tau,
                           practical='Armijo (c=1/4), two-sided (T_E) eta=1/2 and (T_A) c_R=1',
                           tight='Newton-PCG (CG rel. tol 1e-2), ||R||_B<=1e-10', snap_times=times, norms=NORMS),
         out, snaps)
    r = out['rec']
    print('fft', sum(r['fft_p']), sum(r['fft_t']), 'max diffL2', max(r['diffL2']), 'delta', r.get('delta_at'), flush=True)


# ----------------------------------------------------------------------------- droplets
DROP = dict(L=2.0, eps=0.04, A=0.0)


def exp_drop_pilot():
    N, tau, T = 256, 0.001, float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    L, eps, A = DROP['L'], DROP['eps'], DROP['A']
    tr = DropletTracker(DROPS, L)
    times = [round(0.25 * k, 4) for k in range(1, int(T / 0.25) + 1)]
    t0 = time.time()
    r = run_single(N, L, eps, A, tau, T, drops_ic(N, L, eps), dict(method='newton', tolB=1e-10, kmax=60), times, track=tr)
    print('pilot done', time.time() - t0, 't_dis', r['t_dis'], flush=True)
    R = np.array(r['hist']['R']); t = np.array(r['hist']['t'])
    for tk in [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        i = np.argmin(abs(t - tk)); print(tk, R[i], r['hist']['ncomp'][i])
    save('drop_pilot', dict(drops=DROPS, N=N, L=L, eps=eps, A=A, T=T, tau=tau, solver='Newton, ||R||_B<=1e-10'),
         dict(t=r['hist']['t'], R=r['hist']['R'], ncomp=r['hist']['ncomp'], Etil=r['hist']['Etil'], t_dis=r['t_dis'],
              nfft=r['nfft']), r['snaps'])


def exp_drop_refine():
    T = float(sys.argv[2]); times = [float(x) for x in sys.argv[3].split(',')]
    L, eps, A = DROP['L'], DROP['eps'], DROP['A']
    runs = [(256, 0.004), (256, 0.002), (256, 0.001), (256, 0.0005), (128, 0.0005), (512, 0.001)]
    out, snaps = {}, {}
    for N, tau in runs:
        t0 = time.time()
        tr = DropletTracker(DROPS, L)
        r = run_single(N, L, eps, A, tau, T, drops_ic(N, L, eps), dict(method='newton', tolB=1e-10, kmax=60), times, track=tr)
        key = f'N{N}_tau{tau:g}'
        Rh = np.array(r['hist']['R']); th = np.array(r['hist']['t'])
        out[key] = dict(N=N, tau=tau, nfft=r['nfft'], fails=r['fails'], t_dis=r['t_dis'],
                        R_at={f'{tk:g}': Rh[np.argmin(abs(th - tk))].tolist() for tk in times},
                        t=th.tolist(), R=Rh.tolist())
        for k, v in r['snaps'].items():
            snaps[f'{key}_{k}'] = v
        print(key, 'done', time.time() - t0, 't_dis', r['t_dis'], 'R_at', out[key]['R_at'], flush=True)
        save('drop_refine', dict(drops=DROPS, L=L, eps=eps, A=A, T=T, snap_times=times, solver='Newton, ||R||_B<=1e-10',
                                 radius='(A_j/pi)^1/2, A_j enclosed by the marching-squares zero contour'),
             dict(runs=out), snaps)


def exp_drop_prod():
    N, tau, T = int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
    times = [float(x) for x in sys.argv[5].split(',')]
    L, eps, A = DROP['L'], DROP['eps'], DROP['A']
    out, snaps = run_lockstep(N, L, eps, A, tau, T, drops_ic(N, L, eps), times, drops=DROPS)
    save('drop_prod', dict(drops=DROPS, N=N, L=L, eps=eps, A=A, T=T, tau=tau, snap_times=times,
                           practical='Armijo (c=1/4), two-sided (T_E) eta=1/2 and (T_A) c_R=1',
                           tight='Newton-PCG (CG rel. tol 1e-2), ||R||_B<=1e-10',
                           radius='(A_j/pi)^1/2, A_j enclosed by the marching-squares zero contour nearest the previous centroid',
                           norms=NORMS), out, snaps)
    r = out['rec']
    print('fft', sum(r['fft_p']), sum(r['fft_t']), 't_dis', out['t_dis_p'], out['t_dis_t'], 'max diffL2', max(r['diffL2']))


# ----------------------------------------------------------------------------- supplementary seed study
SEED = dict(L=6.4, eps=0.04, A=0.0, mean=-0.5, K0=24, amp=0.05)


def seed_coeffs(seed, K0=SEED['K0']):
    """Random Fourier coefficients c_k, |k_1|,|k_2| <= K0, Hermitian (real field), mean mode zero."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal((2 * K0 + 1, 2 * K0 + 1)) + 1j * rng.standard_normal((2 * K0 + 1, 2 * K0 + 1))
    c = 0.5 * (c + np.conj(c[::-1, ::-1])); c[K0, K0] = 0.0
    return c


def seed_ic(N, seed, mean=SEED['mean'], amp=SEED['amp'], K0=SEED['K0']):
    """phi0 = mean + amp * P(x)/max|P|, P the trigonometric polynomial with the stored coefficients; the same
    function on every grid with N > 2 K0 (the scaling uses the maximum on the N = 1024 grid)."""
    c = seed_coeffs(seed, K0)

    def evaluate(M):
        F = np.zeros((M, M), complex)
        k = np.arange(-K0, K0 + 1)
        F[np.ix_(k % M, k % M)] = c
        return np.real(np.fft.ifft2(F)) * M * M
    scale = np.abs(evaluate(1024)).max()
    return mean + amp * evaluate(N) / scale


def exp_seed_pilot():
    N, tau, T = 256, 0.002, float(sys.argv[2])
    L, eps, A = SEED['L'], SEED['eps'], SEED['A']
    r = run_single(N, L, eps, A, tau, T, seed_ic(N, 1), dict(method='newton', tolB=1e-10, kmax=60), [0.5, 1, 2, 3, 5, 8])
    h = r['hist']; t = np.array(h['t'])
    for tk in [0.25, 0.5, 1, 2, 3, 5, 8]:
        i = np.argmin(abs(t - tk))
        if i < len(t): print(tk, h['ncomp'][i], h['Etil'][i], h['it'][i])
    np.savez_compressed(DATA + 'seed_pilot_snaps.npz', **r['snaps'])


SEED_TIMES = [0.5, 1.0, 2.0, 5.0]


def exp_seeds():
    """Supplementary robustness study: practical vs tight in lockstep for seeds 1, 2, 3 (mean -0.5)."""
    N, tau, T = 256, 0.002, 5.0
    L, eps, A = SEED['L'], SEED['eps'], SEED['A']
    out, snaps = {}, {}
    for seed in [1, 2, 3]:
        t0 = time.time()
        np.save(DATA + f'seed{seed}_coeffs.npy', seed_coeffs(seed))
        o, sn = run_lockstep(N, L, eps, A, tau, T, seed_ic(N, seed), SEED_TIMES)
        r = o['rec']
        out[f'seed{seed}'] = o
        for k, v in sn.items():
            snaps[f'seed{seed}_{k}'] = v.astype(np.float32)
        print('seed', seed, time.time() - t0, 'fft', sum(r['fft_p']), sum(r['fft_t']), 'fails', r['acc_p'].count(False),
              'max diffL2', max(r['diffL2']), 'ncomp p/t at end', r['ncomp_p'][-1], r['ncomp_t'][-1], flush=True)
        save('seeds', dict(L=L, N=N, eps=eps, A=A, tau=tau, T=T, mean=SEED['mean'], amp=SEED['amp'], K0=SEED['K0'],
                           seeds=[1, 2, 3], snap_times=SEED_TIMES,
                           initial='mean + amp*P/max|P|, P trig. polynomial with stored coefficients seed<k>_coeffs.npy',
                           practical='Armijo (c=1/4), two-sided (T_E) eta=1/2 and (T_A) c_R=1',
                           tight='Newton-PCG (CG rel. tol 1e-2), ||R||_B<=1e-10', norms=NORMS), dict(runs=out), snaps)


def exp_seed_refine():
    """Seed 1: tight solves with tau/2 (N=256) and with N=512 (tau=0.002) for the resolution check."""
    L, eps, A, T = SEED['L'], SEED['eps'], SEED['A'], 5.0
    out, snaps = {}, {}
    for N, tau in [(256, 0.001), (512, 0.002)]:
        t0 = time.time()
        r = run_single(N, L, eps, A, tau, T, seed_ic(N, 1), dict(method='newton', tolB=1e-10, kmax=60), SEED_TIMES)
        key = f'N{N}_tau{tau:g}'
        out[key] = dict(N=N, tau=tau, nfft=r['nfft'], fails=r['fails'], t=r['hist']['t'], ncomp=r['hist']['ncomp'])
        for k, v in r['snaps'].items():
            snaps[f'{key}_{k}'] = v.astype(np.float32)
        print(key, time.time() - t0, flush=True)
        save('seed_refine', dict(seed=1, L=L, eps=eps, A=A, T=T, snap_times=SEED_TIMES, solver='Newton, ||R||_B<=1e-10'),
             dict(runs=out), snaps)


if __name__ == '__main__':
    globals()['exp_' + sys.argv[1]]()
