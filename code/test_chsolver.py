import numpy as np, time
from chsolver import CH, chi, Gpot, mms_problem
rng = np.random.default_rng(0)
# setup a coarsening-type step
S = CH(64, 1.6, 0.02, 1.0, 0.004)
phim = 0.8*np.tanh(np.sin(2*np.pi*S.X/1.6)*3) + 0.05*rng.standard_normal((64,64))
phi = phim + 0.01*rng.standard_normal((64,64))
# 1. gradient check: J(v+td)-J(v) ~ t (R,d)
A, eps, tau, s = S.A, S.eps, S.tau, S.s
phih, phimh = np.fft.rfft2(phi), np.fft.rfft2(phim)
fh = phih/tau + s*((2*A+1.5)*phih - (A+0.5)*phimh) - 0.25*eps**2*s**2*phimh
def J(v):
    vh = np.fft.rfft2(v)
    return S.h**2*np.sum(Gpot(v,phi)) + 0.5*S.ipF(vh,vh,S.Binv) - S.ipF(fh,vh,S.invs)
def R(v):
    vh = np.fft.rfft2(v); chih = np.fft.rfft2(chi(v,phi))
    return np.fft.irfft2(S.Binv*vh - S.invs*fh + chih*S.pos, s=(64,64))
v = 2*phi-phim; d = rng.standard_normal((64,64)); d -= d.mean()
for t in [1e-3,1e-4,1e-5]:
    print('grad check', t, (J(v+t*d)-J(v-t*d))/(2*t), S.ip(R(v),d))
# 2. solvers converge to same root
out = {}
for m, kw in [('relax', dict(omega=1.0)), ('relax', dict(omega=0.5)), ('armijo', dict(omega=1.0)), ('newton', {})]:
    S.nfft = 0
    vv, info = S.step(phi, phim, 0, method=m, tolB=1e-11, kmax=500, record=True, **kw)
    out[m+str(kw)] = vv
    print(m, kw, 'iters', info['iters'], 'acc', info['accepted'], 'nfft', S.nfft, 'RB', info['hist'][-1]['RB'])
ref = out['newton{}']
for k,vv in out.items(): print(k, 'diff to newton', np.abs(vv-ref).max())
# 3. perturbed identity for truncated iterate (eta test machinery)
for K in [1,3]:
    vv,_ = S.step(phi, phim, 0, method='fixed', K=K)
    # identity: Et(v,phi) - Et(phi,phim) + D_n = (R(v), v-phi)
    dd = vv-phi; d2 = dd-(phi-phim); ddh=np.fft.rfft2(dd); d2h=np.fft.rfft2(d2)
    Dn = S.ipF(ddh,ddh,S.invs)/tau + (0.25+A/2)*S.ip(d2,d2) + eps**2/8*S.ipF(d2h,d2h,S.s)
    lhs = S.mod_energy(vv,phi) - S.mod_energy(phi,phim) + Dn
    print('identity K=%d'%K, lhs, S.ip(R(vv),dd))
# 4. MMS temporal order, fixed K=1 omega=1
exact, forcing = mms_problem(0.1)
for ns in [32,64,128]:
    P = CH(32, 1.0, 0.1, 1.0, 1/ns, forcing)
    pm, p = exact(P.X,P.Y,-1/ns), exact(P.X,P.Y,0)
    for n in range(ns):
        new,_ = P.step(p, pm, n/ns, method='fixed', K=1); pm, p = p, new
    print('mms', ns, np.abs(p-exact(P.X,P.Y,1.0)).max())
