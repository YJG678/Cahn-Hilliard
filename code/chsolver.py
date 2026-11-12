"""chsolver: second-order modified Crank--Nicolson Fourier pseudo-spectral scheme for the
Cahn--Hilliard equation (Cheng--Wang--Wise--Yue 2016 with alpha = 0) and inner solvers.

    (phi^{n+1}-phi^n)/tau = Delta_N mu^{n+1/2} (+ g^{n+1/2}),
    mu^{n+1/2} = chi(phi^{n+1},phi^n) - (3/2 phi^n - 1/2 phi^{n-1})
                 - eps^2 Delta_N (3/4 phi^{n+1} + 1/4 phi^{n-1}) + A (phi^{n+1} - 2 phi^n + phi^{n-1}),
    chi(a,b) = (a^2+b^2)(a+b)/4.

Inner solvers for the step equation  R_n(v) = 0  (R_n = gradient of a strongly convex J_n):
  'fixed'  : v <- v - omega * B R(v), fixed number of iterations K (omega=1 is the CWWY linear iteration)
  'relax'  : same, fixed omega, stopped by acceptance tests
  'armijo' : same with backtracking on J_n, sufficient decrease J(v-wp) <= J(v) - c_arm w ||R||_B^2
  'newton' : damped Newton, Newton system solved by CG preconditioned with B
Acceptance tests (any combination):
  tolB   : ||R(v)||_B <= tolB                      (a posteriori bound: ||v-phi*||_{B^-1} <= ||R||_B)
  cR     : ||R(v)||_2 <= cR * tau^2                (accuracy test used in the error theorem)
  eta    : |(R(v), v-phi^n)| <= eta * D_n(v)       (energy-safe test; unforced problems only)
All norms are discrete (h^2-weighted).  Every forward/backward real FFT is counted in self.nfft.
"""
import numpy as np


def chi(a, b):
    return 0.25 * (a * a + b * b) * (a + b)


def dchi(a, b):
    """d/da chi(a,b) = (3a^2 + 2ab + b^2)/4 >= 0"""
    return 0.25 * (3 * a * a + 2 * a * b + b * b)


def Gpot(a, b):
    """antiderivative in a of chi(.,b), Gpot(0,b)=0"""
    return a**4 / 16 + a**3 * b / 12 + a * a * b * b / 8 + a * b**3 / 4


class CH:
    def __init__(self, N, L, eps, A, tau, forcing=None):
        self.N, self.L, self.eps, self.A, self.tau = N, L, eps, A, tau
        self.h = L / N
        kx = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        ky = 2 * np.pi * np.fft.rfftfreq(N, d=L / N)
        KX, KY = np.meshgrid(kx, ky, indexing='ij')
        self.s = KX**2 + KY**2                       # symbol of -Delta_N
        self.pos = self.s > 0
        self.Lsym = 1 / tau + A * self.s + 0.75 * eps**2 * self.s**2
        self.Bsym = np.where(self.pos, self.s / self.Lsym, 0.0)     # B = L^{-1}(-Delta)
        self.Binv = np.where(self.pos, self.Lsym / np.where(self.pos, self.s, 1), 0.0)
        self.invs = np.where(self.pos, 1 / np.where(self.pos, self.s, 1), 0.0)
        w = np.full(self.s.shape, 2.0); w[:, 0] = 1.0
        if N % 2 == 0:
            w[:, -1] = 1.0
        self.w = w
        self.c = L**2 / N**4                          # (u,v)_N = c * sum_k u^ conj(v^)
        self.theta_N = float(np.max(self.Bsym))
        x = np.arange(N) * self.h
        self.X, self.Y = np.meshgrid(x, x, indexing='ij')
        self.forcing = forcing
        self.nfft = 0

    # ---- transforms and Fourier-side inner products ----
    def fft(self, u):
        self.nfft += 1
        return np.fft.rfft2(u)

    def ifft(self, uh):
        self.nfft += 1
        return np.fft.irfft2(uh, s=(self.N, self.N))

    def ipF(self, uh, vh, sym=None):
        m = self.w if sym is None else self.w * sym
        return self.c * float(np.sum(m * (uh * np.conj(vh)).real))

    def ip(self, u, v):
        return self.h**2 * float(np.sum(u * v))

    # ---- energies (unforced problem) ----
    def energy(self, phi, phih=None):
        if phih is None:
            phih = np.fft.rfft2(phi)
        return self.h**2 * float(np.sum(0.25 * phi**4 - 0.5 * phi**2)) + 0.5 * self.eps**2 * self.ipF(phih, phih, self.s)

    def mod_energy(self, phi, phim):
        d = phi - phim
        dh = np.fft.rfft2(d)
        return self.energy(phi) + (0.25 + self.A / 2) * self.ip(d, d) + self.eps**2 / 8 * self.ipF(dh, dh, self.s)

    # ---- one time step ----
    def step(self, phi, phim, t, method='relax', omega=1.0, K=None, kmax=200, tolB=None, cR=None, eta=None,
             cg_tol=1e-2, newton_maxcg=50, v0=None, record=False, c_arm=0.25):
        A, eps, tau, s = self.A, self.eps, self.tau, self.s
        phih = self.fft(phi); phimh = self.fft(phim)
        fh = phih / tau + s * ((2 * A + 1.5) * phih - (A + 0.5) * phimh) - 0.25 * eps**2 * s**2 * phimh
        if self.forcing is not None:
            fh = fh + self.fft(self.forcing(self.X, self.Y, t + 0.5 * tau))
        # pieces for the energy-safe test
        d_old_h = phih - phimh
        v = 2 * phi - phim if v0 is None else v0.copy()
        vh = 2 * phih - phimh if v0 is None else self.fft(v0)
        if abs(vh[0, 0] - phih[0, 0]) > 1e-14 * self.N**2:     # enforce the mass of phi^n
            vh[0, 0] = phih[0, 0]
            v = self.ifft(vh)

        def resid(v, vh):
            chih = self.fft(chi(v, phi))
            Gh = (fh - s * chih) / self.Lsym
            ph = vh - Gh                                  # p = v - G(v) = B R(v)
            return chih, Gh, ph

        def dJ(v, R_dot_d, dd, ddh):
            # exact increment J(v+dd) - J(v) without cancellation (J is quartic):
            #   (R(v),dd) + 1/2||dd||_{B^-1}^2 + sum h^2 [dd^2/2 G'' + dd^3/6 G''' + dd^4/16]
            g2 = dchi(v, phi); g3 = 0.25 * (6 * v + 2 * phi)
            loc = self.h**2 * float(np.sum(dd * dd * (0.5 * g2 + dd * (g3 / 6 + dd / 16))))
            return R_dot_d + 0.5 * self.ipF(ddh, ddh, self.Binv) + loc

        def tests(vh, ph):
            Rh = self.Binv * ph
            RB = np.sqrt(max(self.ipF(ph, ph, self.Binv), 0.0))       # ||R||_B = ||p||_{B^-1}
            R2 = np.sqrt(max(self.ipF(Rh, Rh), 0.0))
            dh = vh - phih
            d2h = dh - d_old_h
            Dn = (self.ipF(dh, dh, self.invs) / tau + (0.25 + A / 2) * self.ipF(d2h, d2h)
                  + eps**2 / 8 * self.ipF(d2h, d2h, s))
            pert = abs(self.ipF(Rh, dh))
            ok = True
            if tolB is not None:
                ok &= RB <= tolB
            if cR is not None:
                ok &= R2 <= cR * tau**2
            if eta is not None:
                ok &= pert <= eta * Dn
            rec = dict(RB=float(RB), R2=float(R2), nfft=self.nfft, pert=float(pert), Dn=float(Dn))
            return ok, rec

        hist = []
        it = 0
        accepted = False
        om = omega
        if method in ('fixed', 'relax', 'armijo'):
            chih, Gh, ph = resid(v, vh)
            nmax = K if method == 'fixed' else kmax
            for it in range(nmax + 1):
                if method != 'fixed':
                    ok, rec = tests(vh, ph)
                    if record:
                        hist.append(rec)
                    if ok:
                        accepted = True
                        break
                    if not np.isfinite(rec['RB']) or rec['RB'] > 1e8:      # divergent inner iteration
                        break
                elif record:
                    ok, rec = tests(vh, ph); hist.append(rec)
                if it == nmax:
                    break
                if method == 'armijo':
                    gn2 = self.ipF(ph, ph, self.Binv)            # ||p||^2_{B^-1} = ||R||_B^2 = (R,p)
                    p = self.ifft(ph)
                    om = min(1.0, 2 * om) if it > 0 else om
                    while True:
                        if dJ(v, -om * gn2, -om * p, -om * ph) <= -c_arm * om * gn2 or om < 1e-8:
                            break
                        om *= 0.5
                    vh = vh - om * ph
                    v = v - om * p
                else:
                    vh = vh - om * ph
                    v = self.ifft(vh)
                chih, Gh, ph = resid(v, vh)
            if method == 'fixed':
                accepted = True
        elif method == 'newton':
            chih, Gh, ph = resid(v, vh)
            for it in range(kmax + 1):
                ok, rec = tests(vh, ph)
                if record:
                    hist.append(rec)
                if ok:
                    accepted = True
                    break
                if it == kmax:
                    break
                # Newton system (B^{-1} + P0 D) dv = -R, PCG with preconditioner B, in Fourier variables
                Dv = dchi(v, phi)
                Rh = self.Binv * ph
                bh = -Rh
                xh = np.zeros_like(bh)
                rh = bh.copy()
                zh = self.Bsym * rh
                dh_ = zh.copy()
                rz = self.ipF(rh, zh)
                r0 = np.sqrt(rz)
                for j in range(newton_maxcg):
                    Adh = self.Binv * dh_ + self.fft(Dv * self.ifft(dh_)) * self.pos
                    alpha = rz / self.ipF(dh_, Adh)
                    xh = xh + alpha * dh_
                    rh = rh - alpha * Adh
                    zh = self.Bsym * rh
                    rz_new = self.ipF(rh, zh)
                    if np.sqrt(rz_new) <= cg_tol * r0:
                        break
                    dh_ = zh + (rz_new / rz) * dh_
                    rz = rz_new
                # damped step with Armijo on J
                slope = self.ipF(Rh, xh)
                x = self.ifft(xh)
                om = 1.0
                while True:
                    if dJ(v, om * slope, om * x, om * xh) <= 0.25 * om * slope or om < 1e-8:
                        break
                    om *= 0.5
                vh = vh + om * xh
                v = v + om * x
                chih, Gh, ph = resid(v, vh)
        else:
            raise ValueError(method)
        return v, dict(iters=it, accepted=accepted, hist=hist, omega=om)


def mms_problem(eps=0.1):
    """Manufactured solution phi = sin(2 pi x) cos(2 pi y) cos t on (0,1)^2 (from AC_CH_2nd_1.f)."""
    tp = 2 * np.pi

    def exact(X, Y, t):
        return np.sin(tp * X) * np.cos(tp * Y) * np.cos(t)

    def forcing(X, Y, t):
        sxy = np.sin(tp * X) * np.cos(tp * Y)
        px = tp * np.cos(tp * X) * np.cos(tp * Y)
        py = -tp * np.sin(tp * X) * np.sin(tp * Y)
        lap_s = -2 * tp**2 * sxy
        lap_s3 = 3 * sxy * sxy * lap_s + 6 * sxy * (px**2 + py**2)
        c = np.cos(t)
        return -np.sin(t) * sxy - lap_s3 * c**3 + lap_s * c + eps**2 * (4 * tp**4) * sxy * c
    return exact, forcing


def mms2_problem(eps=0.1, amp=0.4):
    """Non-band-limited manufactured solution phi = amp cos(t) sin(2 pi x) exp(cos(2 pi y)) on (0,1)^2
    (mean zero for all t); forcing generated symbolically."""
    import sympy as sp
    x, y, t = sp.symbols('x y t', real=True)
    u = amp * sp.cos(t) * sp.sin(2 * sp.pi * x) * sp.exp(sp.cos(2 * sp.pi * y))
    lap = lambda f: sp.diff(f, x, 2) + sp.diff(f, y, 2)
    g = sp.diff(u, t) - lap(u**3) + lap(u) + eps**2 * lap(lap(u))
    fu = sp.lambdify((x, y, t), u, 'numpy')
    fg = sp.lambdify((x, y, t), g, 'numpy')
    return (lambda X, Y, tt: fu(X, Y, tt) + 0 * X), (lambda X, Y, tt: fg(X, Y, tt) + 0 * X)
