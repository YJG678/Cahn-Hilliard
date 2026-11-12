"""Experiments for the paper. Usage: python3 experiments.py <name>; results -> ../data/<name>.json"""
import sys, json, time
import numpy as np
from chsolver import CH, mms_problem, mms2_problem

DATA = '../data/'


def hm1(S, e):
    eh = np.fft.rfft2(e)
    return np.sqrt(S.ipF(eh, eh, S.invs))


def run_mms(N, tau, T, A, eps, solver, problem=None):
    exact, forcing = mms_problem(eps) if problem is None else problem
    S = CH(N, 1.0, eps, A, tau, forcing)
    ns = int(round(T / tau))
    pm, p = exact(S.X, S.Y, -tau), exact(S.X, S.Y, 0.0)
    eH = 0.0; iters = 0
    for n in range(ns):
        new, info = S.step(p, pm, n * tau, **solver)
        iters += info['iters']
        pm, p = p, new
        eH = max(eH, hm1(S, p - exact(S.X, S.Y, (n + 1) * tau)))
    e = p - exact(S.X, S.Y, T)
    return dict(tau=tau, N=N, Linf=float(np.abs(e).max()), L2=float(np.sqrt(S.ip(e, e))), Hm1max=float(eH),
                iters=iters / ns, nfft=S.nfft / ns)


def exp_time():
    variants = {
        'exact': dict(method='newton', tolB=1e-12, kmax=50),
        'w1_K1': dict(method='fixed', omega=1.0, K=1),
        'w1_K2': dict(method='fixed', omega=1.0, K=2),
        'w05_K1': dict(method='fixed', omega=0.5, K=1),
        'w05_K2': dict(method='fixed', omega=0.5, K=2),
        'w05_K4': dict(method='fixed', omega=0.5, K=4),
        'w05_acc': dict(method='relax', omega=0.5, cR=1.0, kmax=200),
        'armijo_acc': dict(method='armijo', omega=1.0, cR=1.0, kmax=200),
    }
    out = {}
    for name, sol in variants.items():
        rows = []
        for j in range(4, 11):
            r = run_mms(48, 2.0**-j, 1.0, 1.0, 0.1, sol)
            rows.append(r); print(name, r, flush=True)
        out[name] = rows
    json.dump(out, open(DATA + 'time.json', 'w'), indent=1)


def exp_space():
    rows = []
    prob = mms2_problem(0.1)
    for N in [8, 12, 16, 20, 24, 32, 40, 48, 64]:
        r = run_mms(N, 1e-4, 0.01, 1.0, 0.1, dict(method='newton', tolB=1e-12, kmax=50), problem=prob)
        rows.append(r); print(r, flush=True)
    json.dump(rows, open(DATA + 'space.json', 'w'), indent=1)


def exp_time2():
    out = {}
    for name, sol in {'armijo_acc01': dict(method='armijo', omega=1.0, cR=0.1, kmax=200),
                      'w1_K6': dict(method='fixed', omega=1.0, K=6)}.items():
        rows = []
        for j in range(4, 11):
            r = run_mms(48, 2.0**-j, 1.0, 1.0, 0.1, sol)
            rows.append(r); print(name, r, flush=True)
        out[name] = rows
    json.dump(out, open(DATA + 'time2.json', 'w'), indent=1)


def random_ic(N, seed=1, amp=0.1):
    rng = np.random.default_rng(seed)
    return amp * (2 * rng.random((N, N)) - 1)


def run_coarsen(N, L, eps, A, tau, T, solver, seed=1, keep_energy=False, snaps=(), phi0=None, maxfail=20):
    S = CH(N, L, eps, A, tau)
    phi = random_ic(N, seed) if phi0 is None else phi0.copy()
    phim = phi.copy()
    ns = int(round(T / tau))
    its, fails, Etil, E, ts, snap = [], 0, [], [], [], {}
    Em_prev = S.mod_energy(phi, phim); incr = 0
    t0 = time.time()
    status = 'ok'
    for n in range(ns):
        new, info = S.step(phi, phim, n * tau, **solver)
        its.append(info['iters'])
        if not info['accepted']:
            fails += 1
        if not np.all(np.isfinite(new)) or np.abs(new).max() > 1e3:
            status = 'diverged'; break
        if fails > maxfail:
            status = 'too many failures'; break
        phim, phi = phi, new
        t = (n + 1) * tau
        Em = S.mod_energy(phi, phim)
        if Em > Em_prev + 1e-12 * abs(Em_prev):
            incr += 1
        Em_prev = Em
        if keep_energy:
            Etil.append(Em); E.append(S.energy(phi)); ts.append(t)
        for ts_ in snaps:
            if abs(t - ts_) < 0.5 * tau:
                snap[str(ts_)] = phi.copy()
    wall = time.time() - t0
    res = dict(N=N, L=L, eps=eps, A=A, tau=tau, T=T, solver={k: v for k, v in solver.items()}, status=status,
               steps=len(its), mean_it=float(np.mean(its)), max_it=int(np.max(its)), fails=fails,
               nfft=S.nfft, nfft_per_step=S.nfft / max(len(its), 1), wall=wall, modE_increases=incr,
               theta_N=S.theta_N, Efinal=float(S.energy(phi)) if status == 'ok' else None)
    if keep_energy:
        res.update(t=ts, E=E, Etil=Etil)
    return res, phi, snap


def exp_solver():
    out = []
    methods = {'linear_w1': dict(method='relax', omega=1.0), 'relax_w05': dict(method='relax', omega=0.5),
               'armijo': dict(method='armijo', omega=1.0), 'newton': dict(method='newton')}
    stops = {'tight': dict(tolB=1e-10, kmax=300), 'practical': dict(cR=1.0, eta=0.5, kmax=300)}
    which = sys.argv[2] if len(sys.argv) > 2 else 'all'
    for tau in [0.001, 0.004, 0.016]:
        for A in [0.0, 2.0]:
            for mname, m in methods.items():
                for sname, st in stops.items():
                    tag = f'{mname}|{sname}'
                    if which != 'all' and which != sname:
                        continue
                    sol = dict(m); sol.update(st)
                    r, _, _ = run_coarsen(256, 6.4, 0.02, A, tau, 1.0, sol)
                    r['method'] = mname; r['stop'] = sname
                    out.append(r)
                    print(tag, tau, A, {k: r[k] for k in ['status', 'mean_it', 'max_it', 'fails', 'nfft_per_step', 'wall', 'modE_increases']}, flush=True)
                    json.dump(out, open(DATA + f'solver_{which}.json', 'w'), indent=1)


def exp_mesh():
    out = []
    for N in [64, 128, 256, 512]:
        for mname, sol in {'linear_w1_A2': (2.0, dict(method='relax', omega=1.0)),
                           'armijo_A0': (0.0, dict(method='armijo', omega=1.0)),
                           'newton_A0': (0.0, dict(method='newton'))}.items():
            A, m = sol
            m = dict(m); m.update(tolB=1e-10, kmax=300)
            # same continuous initial data on every mesh: smooth random field (low modes only)
            rng = np.random.default_rng(7)
            k = 2 * np.pi * np.fft.fftfreq(N, d=6.4 / N)
            KX, KY = np.meshgrid(k, k, indexing='ij')
            M = (np.abs(np.fft.fftfreq(N) * N)[:, None] <= 16) & (np.abs(np.fft.fftfreq(N) * N)[None, :] <= 16)
            coef = np.zeros((N, N), complex)
            rng2 = np.random.default_rng(11)
            full = rng2.standard_normal((33, 33)) + 1j * rng2.standard_normal((33, 33))
            idx = np.concatenate([np.arange(0, 17), np.arange(N - 16, N)])
            coef[np.ix_(idx, idx)] = full
            phi0 = np.real(np.fft.ifft2(coef)) * N * N
            phi0 = 0.1 * phi0 / np.abs(phi0).max()
            r, _, _ = run_coarsen(N, 6.4, 0.04, A, 0.004, 1.0, m, phi0=phi0)
            r['method'] = mname
            out.append(r)
            print(N, mname, {k: r[k] for k in ['status', 'mean_it', 'max_it', 'fails', 'nfft_per_step', 'wall']}, flush=True)
            json.dump(out, open(DATA + 'mesh.json', 'w'), indent=1)


def exp_sharp():
    """Two-sided a posteriori bound along iterates of the linear iteration (omega=1) and relaxed iteration."""
    N, L, eps, A, tau = 128, 3.2, 0.02, 2.0, 0.004
    S = CH(N, L, eps, A, tau)
    phi = random_ic(N, 3); phim = phi.copy()
    pts = []
    for n in range(250):
        ref, _ = S.step(phi, phim, 0, method='newton', tolB=1e-13, kmax=60)
        if n % 10 == 0:
            for om in [1.0, 0.5]:
                v = 2 * phi - phim
                for k in range(15):
                    vh_ = np.fft.rfft2(v); e = v - ref; eh = np.fft.rfft2(e)
                    eB = np.sqrt(S.ipF(eh, eh, S.Binv))
                    _, info = S.step(phi, phim, 0, method='fixed', K=0, v0=v, record=True)
                    RB = info['hist'][0]['RB']
                    M = max(np.abs(v).max(), np.abs(ref).max(), np.abs(phi).max())
                    q = 1.5 * M**2 * S.theta_N
                    if eB > 1e-11:
                        pts.append(dict(n=n, omega=om, k=k, eB=float(eB), RB=float(RB), q=float(q)))
                    v, _ = S.step(phi, phim, 0, method='fixed', K=1, omega=om, v0=v)
        phim, phi = phi, ref
    json.dump(pts, open(DATA + 'sharp.json', 'w'))
    r = np.array([p['eB'] / p['RB'] for p in pts]); q = np.array([p['RB'] / p['eB'] / (1 + p['q']) for p in pts])
    print('points', len(pts), 'max eB/RB', r.max(), 'min eB/RB', r.min(), 'max RB/((1+q)eB)', q.max())


def exp_coarsen():
    which = sys.argv[2]
    if which == 'practical':
        sol = dict(method='armijo', omega=1.0, cR=1.0, eta=0.5, kmax=300)
    else:
        sol = dict(method='newton', tolB=1e-10, kmax=60)
    r, phi, snap = run_coarsen(512, 12.8, 0.02, 0.0, 0.004, 10.0, sol, keep_energy=True, snaps=(0.2, 1.0, 4.0, 10.0))
    np.savez_compressed(DATA + f'coarsen_{which}_snaps.npz', **{k: v.astype(np.float32) for k, v in snap.items()})
    json.dump(r, open(DATA + f'coarsen_{which}.json', 'w'))
    print({k: r[k] for k in ['status', 'mean_it', 'max_it', 'fails', 'nfft_per_step', 'wall', 'modE_increases']})


def exp_armijo():
    """Re-run every Armijo configuration (sufficient-decrease constant c = 1/4, the default in chsolver)."""
    part = sys.argv[2]
    if part == 'time':
        out = {}
        for name, cR in [('armijo_acc', 1.0), ('armijo_acc01', 0.1)]:
            rows = [run_mms(48, 2.0**-j, 1.0, 1.0, 0.1, dict(method='armijo', omega=1.0, cR=cR, kmax=200)) for j in range(4, 11)]
            out[name] = rows; print(name, [r['Hm1max'] for r in rows], flush=True)
        json.dump(out, open(DATA + 'time_armijo.json', 'w'), indent=1)
    elif part in ('tight', 'practical'):
        st = dict(tolB=1e-10, kmax=300) if part == 'tight' else dict(cR=1.0, eta=0.5, kmax=300)
        out = []
        for tau in [0.001, 0.004, 0.016]:
            for A in [0.0, 2.0]:
                sol = dict(method='armijo', omega=1.0); sol.update(st)
                r, _, _ = run_coarsen(256, 6.4, 0.02, A, tau, 1.0, sol)
                r['method'] = 'armijo'; r['stop'] = part; out.append(r)
                print(part, tau, A, {k: r[k] for k in ['status', 'mean_it', 'max_it', 'nfft_per_step', 'modE_increases']}, flush=True)
                json.dump(out, open(DATA + f'solver_armijo_{part}.json', 'w'), indent=1)
    elif part == 'mesh':
        out = [r for r in json.load(open(DATA + 'mesh.json')) if r['method'] != 'armijo_A0']
        for N in [64, 128, 256, 512]:
            rng2 = np.random.default_rng(11)
            coef = np.zeros((N, N), complex)
            full = rng2.standard_normal((33, 33)) + 1j * rng2.standard_normal((33, 33))
            idx = np.concatenate([np.arange(0, 17), np.arange(N - 16, N)])
            coef[np.ix_(idx, idx)] = full
            phi0 = np.real(np.fft.ifft2(coef)) * N * N
            phi0 = 0.1 * phi0 / np.abs(phi0).max()
            r, _, _ = run_coarsen(N, 6.4, 0.04, 0.0, 0.004, 1.0, dict(method='armijo', omega=1.0, tolB=1e-10, kmax=300), phi0=phi0)
            r['method'] = 'armijo_A0'; out.append(r)
            print(N, {k: r[k] for k in ['status', 'mean_it', 'max_it', 'nfft_per_step']}, flush=True)
            json.dump(out, open(DATA + 'mesh.json', 'w'), indent=1)


if __name__ == '__main__':
    globals()['exp_' + sys.argv[1]]()
