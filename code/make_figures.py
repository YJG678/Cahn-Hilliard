"""Figures 1-5 and Tables 1-2 of the paper, built only from saved data in ../data.

Plotting standard (one for all figures):
  * figures are drawn at their final printed width (TW = 6.2 in, the text width), so the font sizes set here
    (8-9 pt) are the printed sizes; include them with width=\\textwidth;
  * colour encodes the inner method, line style the stopping rule (solid: residual-based test,
    dashed: fixed number of sweeps), marker shape the individual series; every series has colour AND marker;
  * signed phase fields: diverging map centred at 0 with limits shared by all snapshots;
    nonnegative error magnitudes: sequential map with a shared logarithmic scale;
  * equal aspect ratio for fields; field layers rasterized (300 dpi), curves and text vector.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

D = '../data/'; F = '../figs/'; T = '../tex/'
TW = 6.2
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'cm', 'font.size': 8.5, 'axes.labelsize': 8.5,
                     'axes.titlesize': 8.5, 'legend.fontsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'axes.linewidth': 0.6, 'lines.linewidth': 1.3, 'lines.markersize': 4.5,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
                     'savefig.pad_inches': 0.03, 'legend.handlelength': 2.6, 'legend.frameon': False})
BLUE, ORANGE, AQUA, VIOLET = '#2a78d6', '#eb6834', '#1baf7a', '#6250d6'
INK, MUTED, GRID = '#0b0b0b', '#8a8984', '#d9d8d4'

# method -> (colour, marker, label); stopping rule -> line style / fill (see module docstring)
METH = {'newton': (BLUE, 'o', 'Newton--PCG'), 'armijo': (AQUA, 's', 'Armijo'),
        'relax_w05': (ORANGE, 'v', r'relaxed, $\omega=1/2$'), 'linear_w1': (VIOLET, 'D', r'linear, $\omega=1$')}


def panel(ax, letter, title=''):
    ax.set_title(rf'$\bf({letter})$ ' + title, loc='left', fontsize=8.5, pad=4)


def logticks(ax, axis='x'):
    from matplotlib.ticker import LogLocator, NullFormatter
    a = ax.xaxis if axis == 'x' else ax.yaxis
    a.set_major_locator(LogLocator(base=10)); a.set_minor_formatter(NullFormatter())


def style(m, stop='tight', **kw):
    c, mk, _ = METH[m]
    d = dict(color=c, marker=mk, ls='-' if stop != 'fixed' else '--',
             mfc=(c if stop == 'practical' else 'white'), mec=c, mew=1.0)
    d.update(kw)
    return d


def load_time():
    d = json.load(open(D + 'time.json'))
    for f in ['time2.json', 'time_armijo.json']:
        if os.path.exists(D + f):
            d.update(json.load(open(D + f)))
    return d


# ------------------------------------------------------------------ Figure 1: accuracy
def _accuracy(full, fname):
    d = load_time()
    tau = np.array([r['tau'] for r in d['exact']])
    err = {k: np.array([r['Hm1max'] for r in v]) for k, v in d.items()}
    its = {k: np.array([r['iters'] for r in v]) for k, v in d.items()}
    ex = ('exact', dict(color=BLUE, marker='o', ls='-', mfc=BLUE, mec=BLUE), r'exact solve: Newton, $\|\mathcal{R}_n\|_{\mathcal{B}}\leq10^{-12}$')
    F_all = {'w1_K1': (dict(color=VIOLET, marker='v', ls='--', mfc='white', mec=VIOLET), r'fixed $K=1$ sweep, $\omega=1$'),
             'w1_K6': (dict(color=VIOLET, marker='D', ls='--', mfc='white', mec=VIOLET), r'fixed $K=6$, $\omega=1$ (original code)'),
             'w05_K1': (dict(color=ORANGE, marker='v', ls='--', mfc='white', mec=ORANGE), r'fixed $K=1$, $\omega=1/2$'),
             'w05_K2': (dict(color=ORANGE, marker='^', ls='--', mfc='white', mec=ORANGE), r'fixed $K=2$, $\omega=1/2$'),
             'w05_K4': (dict(color=ORANGE, marker='<', ls='--', mfc='white', mec=ORANGE), r'fixed $K=4$, $\omega=1/2$')}
    R_all = {'w05_acc': (dict(color=ORANGE, marker='o', ls='-', mfc=ORANGE, mec=ORANGE), r'$\omega=1/2$ + (T$_A$), $c_R=1$'),
             'armijo_acc': (dict(color=AQUA, marker='s', ls='-', mfc=AQUA, mec=AQUA), r'Armijo + (T$_A$), $c_R=1$'),
             'armijo_acc01': (dict(color=AQUA, marker='s', ls='-', mfc='white', mec=AQUA), r'Armijo + (T$_A$), $c_R=0.1$')}
    fk = list(F_all) if full else ['w05_K1', 'w1_K6']
    rk = list(R_all) if full else ['w05_acc', 'armijo_acc01']
    fixed = [(k,) + F_all[k] for k in fk]; resid = [(k,) + R_all[k] for k in rk]
    fig, axs = plt.subplots(2, 2, figsize=(TW, 5.0 if full else 4.75),
                            gridspec_kw=dict(hspace=0.45, wspace=0.28, bottom=0.2 if full else 0.19))
    a, b, c, e = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    for ax in (a, b):
        ax.loglog(tau, err['exact'], **ex[1])
        ax.set_xlabel(r'time step $\tau$'); ax.set_ylabel(r'$\max_n\|e^n\|_{-1}$'); ax.set_ylim(8e-8, 3e-3)
        logticks(ax)
        x0, x1 = 2**-10, 2**-8
        ax.loglog([x0, x1], [3e-5, 3e-5 * (x1 / x0)**2], color=MUTED, lw=0.8)
        ax.loglog([x0, x1], [3e-5, 3e-5 * (x1 / x0)], color=MUTED, lw=0.8)
        ax.text(x1 * 1.12, 3e-5 * 16, 'slope 2', color=MUTED, fontsize=7.5, va='center')
        ax.text(x1 * 1.12, 3e-5 * 4, 'slope 1', color=MUTED, fontsize=7.5, va='center')
    for k, st, lab in fixed:
        a.loglog(tau, err[k], **st)
    for k, st, lab in resid:
        b.loglog(tau, err[k], **st)
    panel(a, 'a', 'fixed number of sweeps'); panel(b, 'b', 'residual-based stopping')
    tm = np.sqrt(tau[:-1] * tau[1:])
    order_set = [ex] + ([f for f in fixed if f[0].startswith('w05')] if full else [f for f in fixed if f[0] == 'w05_K1']) + resid
    for k, st, lab in order_set:
        c.semilogx(tm, np.log2(err[k][:-1] / err[k][1:]), **st)
    c.axhline(2, color=MUTED, lw=0.8, zorder=0); c.axhline(1, color=MUTED, lw=0.8, zorder=0)
    c.set_ylim(0.8, 3.0); c.set_xlabel(r'$\sqrt{\tau_{j-1}\tau_j}$'); c.set_ylabel('observed order'); logticks(c)
    panel(c, 'c', 'order between adjacent step sizes')
    for k, st, lab in [ex] + resid:
        e.semilogx(tau, its[k], **st)
    e.set_xlabel(r'time step $\tau$'); e.set_ylabel('inner iterations per step (mean)'); e.set_ylim(0, 4.6); logticks(e)
    panel(e, 'd', 'iterations needed to stop')
    handles = [Line2D([], [], **st, label=lab) for k, st, lab in [ex] + fixed + resid]
    fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.0), columnspacing=1.2)
    fig.savefig(F + fname); plt.close(fig)


def fig1():
    _accuracy(False, 'fig1_accuracy.pdf')


def figS1():
    _accuracy(True, 'figS1_accuracy_all.pdf')


# ------------------------------------------------------------------ Figure 2: robustness and cost
def fig2():
    R = json.load(open(D + 'fig_resid.json'))
    W = json.load(open(D + 'fig_work.json'))
    M = json.load(open(D + 'mesh.json'))
    fig, axs = plt.subplots(2, 2, figsize=(TW, 5.4), gridspec_kw=dict(hspace=0.45, wspace=0.34, top=0.86))
    for ax, key, ttl, letter in [(axs[0, 0], 'A0', '$A=0$', 'a'), (axs[0, 1], 'A2', '$A=2$', 'b')]:
        block = R[key]
        for m in ['linear_w1', 'relax_w05', 'armijo', 'newton']:
            h = block['methods'][m]
            x = np.array(h['nfft']); y = np.array(h['RB'])
            ok = np.isfinite(y) & (y <= 1e7)
            col, mk, lab = METH[m]
            ax.semilogy(x[ok], y[ok], '-', color=col, lw=1.1)
            step = max(1, ok.sum() // 8)
            ax.semilogy(x[ok][::step], y[ok][::step], mk, color=col, mfc='white', mec=col, ms=4)
            pa = h['practical_accept']
            if pa is not None:
                ax.semilogy(x[pa], y[pa], mk, color=col, mfc=col, mec=INK, ms=7, mew=0.9, zorder=5)
            if np.isfinite(y[-1]) and y[-1] > 1e7:          # off-scale divergent iterate
                ax.semilogy([x[ok][-1], x[-1]], [y[ok][-1], 1e7], '-', color=col, lw=1.1, clip_on=False)
                ax.semilogy([x[-1]], [1e7], '^', color=col, ms=6, clip_on=False, zorder=6)
                ax.text(x[-1] + 6, 1e7, r'$%.0f\times10^{%d}$ off scale:' % (y[-1] / 10**np.floor(np.log10(y[-1])), np.floor(np.log10(y[-1])))
                        + '\nstep aborted (divergence)', color=col, fontsize=7.5, va='top')
        ax.axhline(1e-10, color=MUTED, lw=0.8, ls=':')
        ax.set_ylim(1e-13, 1e7)
        ax.set_xlabel('transforms in this time step'); ax.set_ylabel(r'$\|\mathcal{R}_n(v^k)\|_{\mathcal{B}}$')
        panel(ax, letter, f'one hard step ($t=0.08$), {ttl}')
    hs = [Line2D([], [], color=METH[m][0], marker=METH[m][1], mfc='white', mec=METH[m][0], label=METH[m][2])
          for m in ['linear_w1', 'relax_w05', 'armijo', 'newton']]
    hs += [Line2D([], [], ls='', marker='o', color=MUTED, mfc=MUTED, mec=INK, ms=6, label='filled: practical rule (accepts)'),
           Line2D([], [], ls='-', marker='o', color=MUTED, mfc='white', label=r'open: tight rule'),
           Line2D([], [], ls='--', marker='D', color=VIOLET, mfc='white', label=r'dashed: fixed $K=6$, $\omega=1$')]
    fig.legend(handles=hs, loc='lower center', bbox_to_anchor=(0.5, 0.885), ncol=4, fontsize=7.5, handlelength=2.0, columnspacing=1.0)
    # (c) work-precision on the smooth interface-relaxation problem
    ax = axs[1, 0]
    rows = W['rows']; floor = W['meta']['ref_error_L2_estimate']
    series = [('code_K6', VIOLET, 'D', '--', 'white', r'$\omega=1$, $K=6$ (orig. code)'),
              ('linear_tight', VIOLET, 'D', '-', 'white', r'linear $\omega=1$, tight'),
              ('armijo_tight', AQUA, 's', '-', 'white', 'Armijo, tight'), ('armijo_pract', AQUA, 's', '-', AQUA, 'Armijo, practical'),
              ('newton_tight', BLUE, 'o', '-', 'white', 'Newton, tight'), ('newton_pract', BLUE, 'o', '-', BLUE, 'Newton, practical')]
    for k, col, mk, ls, fc, lab in series:
        rr = sorted([r for r in rows if r['method'] == k], key=lambda r: -r['tau'])
        ok = [r for r in rr if not r['diverged']]
        ax.loglog([r['nfft_total'] for r in ok], [r['errL2'] for r in ok], ls=ls, color=col, marker=mk, mfc=fc, mec=col, label=lab)
    ax.axhspan(5e-7, floor, color=GRID, zorder=0, lw=0)
    ax.text(1.4e5, np.sqrt(5e-7 * floor), r'reference error $\approx %.1f\times10^{-6}$' % (floor * 1e6), ha='right', color=INK, fontsize=7.5, va='center')
    ax.set_ylim(5e-7, 0.3); ax.set_xlim(2e2, 1.5e5)
    ax.set_xlabel('total transforms to $T=0.5$'); ax.set_ylabel(r'$\|\phi^{n_T}-\phi_{\mathrm{ref}}(T)\|$')
    for tau in sorted({r['tau'] for r in rows}, reverse=True):
        rr = [r for r in rows if r['tau'] == tau and not r['diverged']]
        x0 = max(r['nfft_total'] for r in rr); y0 = max(r['errL2'] for r in rr)
        ax.text(x0 * 1.3, y0, r'$\tau=%g$' % tau, fontsize=7.5, ha='left', va='center', color=MUTED)
    rr = {r['method']: r for r in rows if r['tau'] == min(r['tau'] for r in rows)}
    ap, k6, nt = rr['armijo_pract'], rr['code_K6'], rr['newton_tight']
    ax.text(2.3e2, 2.0e-5, 'lowest points ($\\tau=%g$): practical Armijo\nuses %.0f%% fewer transforms than $K=6$\nand %.0f%% fewer than tight Newton'
            % (ap['tau'], 100 * (1 - ap['nfft_total'] / k6['nfft_total']), 100 * (1 - ap['nfft_total'] / nt['nfft_total'])),
            fontsize=7.5, ha='left', va='center')
    panel(ax, 'c', 'error vs. work (smooth problem, $A=2$)')
    # (d) transforms per step versus resolution
    ax = axs[1, 1]
    for m, key, lab in [('linear_w1', 'linear_w1_A2', r'linear $\omega=1$, $A=2$'), ('armijo', 'armijo_A0', 'Armijo, $A=0$'),
                        ('newton', 'newton_A0', 'Newton--PCG, $A=0$')]:
        rr = sorted([r for r in M if r['method'] == key], key=lambda r: r['N'])
        col, mk, _ = METH[m]
        ax.semilogx([r['N'] for r in rr], [r['nfft_per_step'] for r in rr], '-', color=col, marker=mk, mfc='white', mec=col, label=lab)
    ax.set_xscale('log', base=2); ax.set_xticks([64, 128, 256, 512]); ax.set_xticklabels(['64', '128', '256', '512'])
    ax.minorticks_off()
    ax.set_ylim(0, 70); ax.set_xlabel('$N$ (grid points per direction)'); ax.set_ylabel('mean transforms per step')
    ax.legend(loc='lower right', fontsize=7.5)
    panel(ax, 'd', r'work vs. resolution ($\tau=0.004$, tight)')
    fig.savefig(F + 'fig2_robustness.pdf'); plt.close(fig)


# ------------------------------------------------------------------ Figure 3: estimator reliability
def fig3():
    pts = json.load(open(D + 'sharp.json'))
    A = json.load(open(D + 'fig_accept.json'))['pts']
    eB = np.array([p['eB'] for p in pts]); RB = np.array([p['RB'] for p in pts]); q = np.array([p['q'] for p in pts])
    om = np.array([p['omega'] for p in pts])
    fig, (a, b) = plt.subplots(1, 2, figsize=(TW, 2.9), gridspec_kw=dict(wspace=0.32, bottom=0.27))
    for w, m in [(1.0, 'linear_w1'), (0.5, 'relax_w05')]:
        col, mk, lab = METH[m]
        s = om == w
        a.semilogx(RB[s], eB[s] / RB[s], mk, color=col, mfc='white', mec=col, ms=3, mew=0.7, label=lab, rasterized=True)
    a.axhline(1.0, color=INK, lw=0.9)
    lb = 1 / (1 + q.max())
    a.axhline(lb, color=MUTED, lw=0.9, ls='--')
    a.text(2e-9, 1.015, 'guaranteed upper bound (a posteriori bound)', fontsize=7.5, va='bottom')
    a.text(1e-2, lb - 0.015, r'lower bound $1/(1+q_{\max})$', fontsize=7.5, color=MUTED, va='top', ha='right')
    a.set_ylim(0.3, 1.12); a.set_xlabel(r'residual $\|\mathcal{R}_n(v)\|_{\mathcal{B}}$')
    a.set_ylabel(r'$\|v-\phi^{n+1}\|_{\mathcal{B}^{-1}}\ /\ \|\mathcal{R}_n(v)\|_{\mathcal{B}}$')
    panel(a, 'a', 'effectivity along inner iterates')
    offs = {'linear_w1': 0.70, 'relax_w05': 0.84, 'armijo': 1.19, 'newton': 1.42}
    for m in ['linear_w1', 'relax_w05', 'armijo', 'newton']:
        col, mk, lab = METH[m]
        pp = [p for p in A if p['method'] == m]
        b.loglog([p['tol'] * offs[m] for p in pp], [p['err'] / p['tol'] for p in pp], mk, color=col, mfc='white', mec=col,
                 ms=3.5, mew=0.7, label=lab)
    b.axhline(1.0, color=INK, lw=0.9)
    b.axhspan(1.0, 10, color=GRID, lw=0, zorder=0)
    b.text(0.5, 3.0, 'excluded by the a posteriori bound', fontsize=7.5, ha='center', va='center', transform=b.get_yaxis_transform())
    b.set_ylim(1e-3, 10); b.set_xlim(2e-11, 3e-2)
    b.set_xlabel(r'tolerance tol in $\|\mathcal{R}_n(v)\|_{\mathcal{B}}\leq$ tol')
    b.set_ylabel(r'$\|v-\phi^{n+1}\|_{\mathcal{B}^{-1}}\ /\ $tol')
    panel(b, 'b', 'true error of accepted iterates / tolerance')
    hs = [Line2D([], [], ls='', marker=METH[m][1], color=METH[m][0], mfc='white', label=METH[m][2])
          for m in ['linear_w1', 'relax_w05', 'armijo', 'newton']]
    fig.legend(handles=hs, loc='upper center', bbox_to_anchor=(0.5, 0.085), ncol=4, fontsize=7.5)
    fig.savefig(F + 'fig3_estimator.pdf'); plt.close(fig)


# ------------------------------------------------------------------ Figures 4 and 5: coarsening benchmark
def fig4():
    S = np.load(D + 'fig_bench_snaps.npz')
    meta = json.load(open(D + 'fig_bench.json'))['meta']
    L = meta['L']; N = meta['N']; h = L / N
    times = ['0.2', '1', '4', '10']
    vlim = 1.2
    dmin, dmax = 1e-6, 1.5e-2
    ztk = '1'                                     # zoom: neighbourhood of the largest difference at t = 1
    dz = np.abs(S['p_' + ztk].astype(float) - S['t_' + ztk].astype(float))
    ic, jc = np.unravel_index(dz.argmax(), dz.shape)
    half = 32                                     # 64 x 64 grid points = 1.6 x 1.6
    i0, j0 = ic - half, jc - half
    box = (i0 * h, j0 * h, 2 * half * h)
    H = 3.2
    w = 0.168; hh = w * TW / H; gap = 0.010; zgap = 0.035
    ytop, ybot = 0.565, 0.135
    fig = plt.figure(figsize=(TW, H))
    xs = np.arange(N) * h

    def put(col, row):
        x = col * (w + gap) + (zgap if col == 4 else 0.0)
        return fig.add_axes([x, ytop if row == 0 else ybot, w, hh])

    def fmt(v):
        e = int(np.floor(np.log10(v)))
        return r'$%.1f\times10^{%d}$' % (v / 10**e, e)

    for i, tk in enumerate(times + ['zoom']):
        tsel = ztk if tk == 'zoom' else tk
        p = S['p_' + tsel].astype(float); t_ = S['t_' + tsel].astype(float)
        if tk == 'zoom':
            sl = (slice(i0, i0 + 2 * half), slice(j0, j0 + 2 * half))
            p, t_ = p[sl], t_[sl]
            ext = [box[0], box[0] + box[2], box[1], box[1] + box[2]]
            xx = box[0] + np.arange(2 * half) * h; yy = box[1] + np.arange(2 * half) * h
        else:
            ext = [0, L, 0, L]; xx = yy = xs
        ax = put(i, 0)
        im1 = ax.imshow(p.T, origin='lower', cmap='RdBu_r', vmin=-vlim, vmax=vlim, extent=ext,
                        interpolation='nearest', rasterized=True)
        ax.contour(xx, yy, t_.T, levels=[0.0], colors=INK, linewidths=0.45 if tk != 'zoom' else 0.9)
        if tk == 'zoom':
            ax.contour(xx, yy, p.T, levels=[0.0], colors='white', linewidths=0.6, linestyles='--')
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(r'zoom, $t=%s$' % ztk if tk == 'zoom' else rf'$t={tk}$', pad=3)
        if i == 0:
            ax.set_ylabel(r'$\phi$ (practical)', fontsize=8.5)
        ax2 = put(i, 1)
        dd = np.abs(p - t_)
        im2 = ax2.imshow(np.clip(dd, dmin, None).T, origin='lower', cmap='viridis', norm=LogNorm(dmin, dmax),
                         extent=ext, interpolation='nearest', rasterized=True)
        ax2.set_aspect('equal'); ax2.set_xticks([]); ax2.set_yticks([])
        fig.text(ax2.get_position().x0 + w / 2, ybot - 0.03, 'max ' + fmt(dd.max()), ha='center', va='top', fontsize=7.5)
        if i == 0:
            ax2.set_ylabel(r'$|\phi_{\mathrm{pract}}-\phi_{\mathrm{tight}}|$', fontsize=8.5)
        if tk == ztk:
            for axx in (ax, ax2):
                axx.add_patch(plt.Rectangle(box[:2], box[2], box[2], fill=False, ec='white', lw=1.6))
                axx.add_patch(plt.Rectangle(box[:2], box[2], box[2], fill=False, ec=INK, lw=0.7))
    xr = 4 * (w + gap) + zgap + w + 0.012
    cax1 = fig.add_axes([xr, ytop + 0.03, 0.012, hh - 0.06])
    cb1 = fig.colorbar(im1, cax=cax1, ticks=[-1, -0.5, 0, 0.5, 1]); cb1.ax.tick_params(labelsize=7.5)
    cax2 = fig.add_axes([xr, ybot + 0.03, 0.012, hh - 0.06])
    cb2 = fig.colorbar(im2, cax=cax2, extend='min'); cb2.ax.tick_params(labelsize=7.5)
    fig.savefig(F + 'fig4_coarsening.pdf'); plt.close(fig)
    return dict(box=[float(v) for v in box], center_grid=(int(ic), int(jc)),
                ranges={tk: (float(S[f'p_{tk}'].min()), float(S[f'p_{tk}'].max())) for tk in times})


def fig5():
    B = json.load(open(D + 'fig_bench.json')); r = B['rec']; meta = B['meta']
    t = np.array(r['t']); area = meta['L']**2
    Etp, Ett = np.array(r['Etil_p']), np.array(r['Etil_t'])
    fig, axs = plt.subplots(2, 2, figsize=(TW, 4.6), gridspec_kw=dict(hspace=0.5, wspace=0.34))
    a, b, c, d = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    a.loglog(t, Etp + area / 4, '-', color=AQUA, label='practical (Armijo)')
    a.loglog(t, Ett + area / 4, '--', color=BLUE, label='tight (Newton--PCG)')
    inc = max(np.diff(Etp).max(), np.diff(Ett).max())
    a.text(0.97, 0.97, (r'$\max_n\,\Delta\widetilde E=%.1f\times10^{%d}$' % (inc / 10**np.floor(np.log10(abs(inc))), np.floor(np.log10(abs(inc)))))
           + ('\n(no increase in either run)' if inc <= 0 else ''),
           transform=a.transAxes, ha='right', va='top', fontsize=7.5)
    a.set_xlabel('$t$'); a.set_ylabel(r'$\widetilde E(\phi^n,\phi^{n-1})+|\Omega|/4$'); logticks(a)
    tt = np.array([1.0, 10.0]); E1 = (Etp + area / 4)[np.argmin(abs(t - 1.0))]
    a.loglog(tt, 0.75 * E1 * tt**(-1 / 3), color=MUTED, lw=0.8)
    a.text(3.2, 0.75 * E1 * 3.2**(-1 / 3) * 0.8, r'$t^{-1/3}$', color=MUTED, fontsize=7.5, va='top')
    a.legend(loc='lower left', fontsize=7.5)
    panel(a, 'a', 'modified energy')
    b.semilogy(t, np.maximum(np.abs(r['mass_p']), 1e-19), '-', color=AQUA, lw=0.8, label='practical')
    b.semilogy(t, np.maximum(np.abs(r['mass_t']), 1e-19), '--', color=BLUE, lw=0.9, label='tight')
    b.axhline(np.finfo(float).eps, color=MUTED, lw=0.8, ls=':')
    b.text(0.12, np.finfo(float).eps / 1.6, r'$\varepsilon_{\mathrm{mach}}=2.2\times10^{-16}$ (unit scale, for reference)', color=MUTED, fontsize=7.5, ha='left', va='top', transform=b.get_yaxis_transform())
    b.set_ylim(1e-19, 1e-12); b.set_xlabel('$t$'); b.set_ylabel(r'$|\mathrm{mean}(\phi^n)-\mathrm{mean}(\phi^0)|$')
    b.legend(loc='upper left', fontsize=7.5)
    panel(b, 'b', 'mass drift')
    c.loglog(t, r['diffL2'], '-', color=VIOLET, label=r'$L^2$ norm')
    c.loglog(t, r['diffH1'], '--', color=ORANGE, label=r'$H^1$ seminorm')
    c.loglog(t, r['diffInf'], ':', color=INK, lw=1.1, label='max. norm')
    c.set_xlabel('$t$'); c.set_ylabel(r'$\|\phi_{\mathrm{pract}}^n-\phi_{\mathrm{tight}}^n\|$'); logticks(c)
    if os.path.exists(D + 'fig_tauscale.npz'):
        H = np.load(D + 'fig_tauscale.npz'); Sn = np.load(D + 'fig_bench_snaps.npz')
        tk = [0.2, 1.0]
        eff = [np.sqrt(((Sn[f't_{x:g}'].astype(float) - H[f'h_{x:g}'])**2).mean()) * meta['L'] for x in tk]
        c.loglog(tk, eff, '*', color=VIOLET, ms=8, mec=INK, mew=0.5, label=r'$\tau$ vs. $\tau/2$ ($L^2$)')
    c.set_ylim(1e-6, 1e2); c.set_xlim(t[0] * 0.8, 70)
    for key, lab in [('diffH1', r'$H^1$ semi.'), ('diffInf', 'max.'), ('diffL2', r'$L^2$')]:
        yv = np.array(r[key])[-200:].mean()
        c.text(12, yv * (1.6 if key == 'diffInf' else 1.0), lab, fontsize=7.5, va='center', ha='left',
               color={'diffL2': VIOLET, 'diffH1': ORANGE, 'diffInf': INK}[key])
    if os.path.exists(D + 'fig_tauscale.npz'):
        c.text(1.5, eff[1], r'$\tau$ vs. $\tau/2$ ($L^2$)', fontsize=7.5, va='center', ha='left', color=VIOLET)
    panel(c, 'c', 'practical versus tight solution')
    ratio = np.maximum(np.array(r['TEratio_p']), 1e-16)
    d.loglog(t, ratio, '.', color=AQUA, ms=1.6, rasterized=True)
    d.axhline(0.5, color=INK, lw=0.9)
    d.text(0.006, 0.7, r'threshold $\eta=1/2$ of test (T$_E$)', fontsize=7.5, va='bottom')
    d.set_ylim(1e-11, 10); d.set_xlabel('$t$'); d.set_ylabel(r'$|(\mathcal{R}_n,\phi^{n+1}-\phi^n)|\,/\,\mathcal{D}_n$'); logticks(d)
    panel(d, 'd', 'two-sided energy test (T$_E$) at acceptance')
    fig.savefig(F + 'fig5_structure.pdf'); plt.close(fig)


# ------------------------------------------------------------------ Figures 6 and 7: interface evolution
def _l2(a, b, L):
    return float(np.sqrt(((a - b)**2).mean()) * L)


def _fmt(v):
    if v == 0:
        return '$0$'
    e = int(np.floor(np.log10(v)))
    return r'$%.1f\times10^{%d}$' % (v / 10**e, e)


def _image_rows(fig, S, times, L, zoom, ybot, ytop, dlim, vlim=1.2, zlabel=None, labels=True, markon=None):
    """Top row: practical phi with tight zero contour; bottom row: |practical - tight| on a shared log scale.
    zoom = (time key, x0, y0, size) for a fifth column."""
    N = S['p_0'].shape[0]; h = L / N; xs = np.arange(N) * h
    w = 0.168; hh = w * fig.get_figwidth() / fig.get_figheight(); gap = 0.010; zgap = 0.035
    zk, zx, zy, zs = zoom
    i0, j0 = int(round(zx / h)), int(round(zy / h)); nz = int(round(zs / h))
    box = (i0 * h, j0 * h, nz * h)
    keys = times + ['zoom']
    for c, tk in enumerate(keys):
        kk = zk if tk == 'zoom' else tk
        p = S['p_' + kk].astype(float); t_ = S['t_' + kk].astype(float)
        if tk == 'zoom':
            sl = (slice(i0, i0 + nz), slice(j0, j0 + nz)); p, t_ = p[sl], t_[sl]
            ext = [box[0], box[0] + box[2], box[1], box[1] + box[2]]; xx = yy = None
            xx = box[0] + np.arange(nz) * h; yy = box[1] + np.arange(nz) * h
        else:
            ext = [0, L, 0, L]; xx = yy = xs
        x = c * (w + gap) + (zgap if tk == 'zoom' else 0.0)
        ax = fig.add_axes([x, ytop, w, hh])
        im1 = ax.imshow(p.T, origin='lower', cmap='RdBu_r', vmin=-vlim, vmax=vlim, extent=ext, interpolation='nearest', rasterized=True)
        ax.contour(xx, yy, t_.T, levels=[0.0], colors=INK, linewidths=0.5 if tk != 'zoom' else 0.9)
        if tk == 'zoom':
            ax.contour(xx, yy, p.T, levels=[0.0], colors='white', linewidths=0.6, linestyles='--')
        ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title((zlabel or r'zoom, $t=%s$' % zk) if tk == 'zoom' else rf'$t={tk}$', pad=3)
        if c == 0 and labels:
            ax.set_ylabel(r'$\phi$ (practical)', fontsize=8.5)
        ax2 = fig.add_axes([x, ybot, w, hh])
        dd = np.abs(p - t_)
        im2 = ax2.imshow(np.clip(dd, dlim[0], None).T, origin='lower', cmap='viridis', norm=LogNorm(*dlim), extent=ext,
                         interpolation='nearest', rasterized=True)
        ax2.set_aspect('equal'); ax2.set_xticks([]); ax2.set_yticks([])
        fig.text(x + w / 2, ybot - 0.012, 'max ' + _fmt(dd.max()) if dd.max() > 0 else 'identical data',
                 ha='center', va='top', fontsize=7.5)
        if c == 0 and labels:
            ax2.set_ylabel(r'$|\phi_{\mathrm{pract}}-\phi_{\mathrm{tight}}|$', fontsize=8.5)
        if tk == (markon or zk):
            for axx in (ax, ax2):
                axx.add_patch(plt.Rectangle(box[:2], box[2], box[2], fill=False, ec='white', lw=1.6))
                axx.add_patch(plt.Rectangle(box[:2], box[2], box[2], fill=False, ec=INK, lw=0.7))
    xr = 4 * (w + gap) + zgap + w + 0.012
    cb1 = fig.colorbar(im1, cax=fig.add_axes([xr, ytop + 0.02, 0.012, hh - 0.04]), ticks=[-1, -0.5, 0, 0.5, 1])
    cb2 = fig.colorbar(im2, cax=fig.add_axes([xr, ybot + 0.02, 0.012, hh - 0.04]), extend='min')
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=7.5)
    return box


def fig6():
    B = json.load(open(D + 'fig_star_prod.json')); r = B['rec']; meta = B['meta']; L = meta['L']
    S = np.load(D + 'fig_star_prod_snaps.npz')
    R = np.load(D + 'fig_star_refine_snaps.npz')
    t = np.array(r['t'])
    # neck of the second merger at t = 0.19: saddle of the tight solution on the symmetry line y = L/2
    N = meta['N']; h = L / N; jm = N // 2
    ph = S['t_0.19'][:, jm].astype(float); i1 = int(0.9 / h)
    while ph[i1] > 0: i1 += 1                       # leave the relaxed star
    i2 = i1
    while ph[i2] < 0: i2 += 1                       # reach the merged pair
    ic = i1 + int(np.argmax(ph[i1:i2]))             # saddle of the neck
    zs = 0.4; zoom = ('0.19', ic * h - zs / 2, L / 2 - zs / 2, zs)
    fig = plt.figure(figsize=(TW, 5.0))
    box = _image_rows(fig, S, ['0', '0.1', '0.25', '0.5'], L, zoom, ybot=0.43, ytop=0.675, dlim=(1e-10, 1e-5),
                      zlabel=r'neck, $t=0.19$', markon='0.25')
    # discrepancy history and time-step effect
    a = fig.add_axes([0.085, 0.075, 0.38, 0.26]); b = fig.add_axes([0.585, 0.075, 0.38, 0.26])
    a.semilogy(t, r['diffL2'], '-', color=VIOLET, lw=1.0)
    a.semilogy(t, r['diffInf'], ':', color=INK, lw=1.0)
    da = r['delta_at']; tk = sorted(float(k) for k in da)
    a.semilogy(tk, [da[f'{x:g}'] for x in tk], 'o', color=ORANGE, mfc='white', ms=4)
    ref_t = [0.1, 0.25, 0.5]
    sp = [_l2(R[f'N128_tau0.001_{x:g}'], R[f'N128_tau0.0005_{x:g}'], L) for x in ref_t]
    from fig_experiments2 import interface_disp
    sd = [interface_disp(R[f'N128_tau0.001_{x:g}'], R[f'N128_tau0.0005_{x:g}'], L) for x in ref_t]
    a.semilogy(ref_t, sp, '*', color=VIOLET, ms=8, mec=INK, mew=0.5)
    a.semilogy(ref_t, sd, '*', color=ORANGE, ms=8, mec=INK, mew=0.5)
    for tm in [0.030, 0.194]:
        a.axvline(tm, color=GRID, lw=0.9, zorder=0)
    a.text(0.036, 0.3, 'merger 1', fontsize=7.5, color=MUTED, va='center')
    a.text(0.200, 0.3, 'merger 2', fontsize=7.5, color=MUTED, va='center')
    a.set_ylim(1e-14, 1); a.set_xlim(0, 0.5); a.set_xlabel('$t$'); a.set_ylabel('difference')
    hs = [Line2D([], [], color=VIOLET, label=r'$L^2$'), Line2D([], [], color=INK, ls=':', label='max.'),
          Line2D([], [], ls='', marker='o', color=ORANGE, mfc='white', label=r'$\delta_\Gamma$'),
          Line2D([], [], color=MUTED, label='pract. vs. tight'),
          Line2D([], [], ls='', marker='*', color=MUTED, mec=INK, ms=8, label=r'$\tau$ vs. $\tau/2$')]
    a.legend(handles=hs, loc='lower right', fontsize=7, ncol=3, handlelength=1.4, columnspacing=0.6, framealpha=0.95)
    a.set_yticks([1e-12, 1e-9, 1e-6, 1e-3, 1])
    panel(a, 'a', r'solver discrepancy vs. time-step effect')
    fp = np.array(r['fft_p']); ft = np.array(r['fft_t'])
    b.plot(t, ft, '-', color=BLUE, lw=0.9); b.plot(t, fp, '-', color=AQUA, lw=0.9)
    for tm in [0.030, 0.194]:
        b.axvline(tm, color=GRID, lw=0.9, zorder=0)
    b.text(0.49, 31, 'tight (Newton--PCG): %d in total' % ft.sum(), color=BLUE, fontsize=7.5, ha='right', va='bottom')
    b.text(0.49, 3, 'practical (Armijo): %d in total' % fp.sum(), color=AQUA, fontsize=7.5, ha='right', va='bottom')
    b.set_ylim(0, 40); b.set_xlim(0, 0.5); b.set_xlabel('$t$'); b.set_ylabel('transforms per step')
    panel(b, 'b', 'cost per step')
    fig.savefig(F + 'fig6_star.pdf'); plt.close(fig)
    return dict(box=box, neck_x=ic * h, tau_effect_L2=sp, tau_effect_delta=sd)


def fig7():
    B = json.load(open(D + 'fig_drop_prod.json')); r = B['rec']; meta = B['meta']; L = meta['L']
    S = np.load(D + 'fig_drop_prod_snaps.npz')
    RF = json.load(open(D + 'fig_drop_refine.json'))['runs']
    t = np.array(r['t']); Rp = np.array(r['R_p']); Rt = np.array(r['R_t'])
    (cx, cy), _ = meta['drops'][0]
    zs = 0.4; zoom = ('0.15', cx - zs / 2, cy - zs / 2, zs)
    fig = plt.figure(figsize=(TW, 5.0))
    box = _image_rows(fig, S, ['0', '0.15', '1', '2'], L, zoom, ybot=0.43, ytop=0.675, dlim=(1e-10, 1e-4),
                      zlabel=r'small droplet, $t=0.15$')
    a = fig.add_axes([0.085, 0.075, 0.38, 0.26]); b = fig.add_axes([0.585, 0.075, 0.38, 0.26])
    cols = [ORANGE, VIOLET, BLUE]; names = ['$R_1$ (small)', '$R_2$ (medium)', '$R_3$ (large)']
    ref = RF['N256_tau0.0005']; tr = np.array(ref['t']); Rr = np.array(ref['R'])
    half = RF['N256_tau0.001']
    for j in range(3):
        a.plot(t, Rp[:, j], '-', color=cols[j], lw=1.2, label=names[j])
        a.plot(t, Rt[:, j], '--', color=INK, lw=0.6)
        for tk in meta['snap_times']:
            i = np.argmin(abs(tr - tk))
            if Rr[i, j] > 0:
                a.plot(tr[i], Rr[i, j], 'o', color=cols[j], mfc='white', ms=4, zorder=5)
        td = B['t_dis_p'][j]
        if td is not None:
            a.axvline(td, color=cols[j], lw=0.6, ls=':')
            a.text(td + 0.03, 0.40 if j == 0 else 0.03, r'$t_%d=%.3f$' % (j + 1, td), color=cols[j], fontsize=7.5,
                   va='center')
    a.set_xlim(0, 2.06); a.set_ylim(0, 0.45); a.set_xlabel('$t$'); a.set_ylabel('equivalent radius $R_j$')
    a.legend(loc='lower left', fontsize=7, handlelength=1.4, bbox_to_anchor=(0.12, 0.02))
    panel(a, 'a', 'droplet radii')
    for j in range(3):
        dR = np.abs(Rp[:, j] - Rt[:, j]); ok = (Rp[:, j] > 0) & (Rt[:, j] > 0)
        b.semilogy(t[ok], np.maximum(dR[ok], 1e-12), '-', color=cols[j], lw=0.9)
        # time-step effect: tight tau = 0.001 vs tau/2 = 0.0005 at every step of the coarser run
        th = np.array(half['t']); Rh = np.array(half['R'])
        idx = np.searchsorted(tr, th - 1e-12)
        ok2 = (Rh[:, j] > 0) & (Rr[idx, j] > 0)
        b.semilogy(th[ok2], np.maximum(np.abs(Rh[ok2, j] - Rr[idx[ok2], j]), 1e-12), ':', color=cols[j], lw=1.1)
    b.set_xlim(0, 2); b.set_ylim(1e-10, 1e-2); b.set_xlabel('$t$'); b.set_ylabel('radius difference')
    hs = [Line2D([], [], color=MUTED, label='practical vs. tight'),
          Line2D([], [], ls=':', color=MUTED, lw=1.1, label=r'tight $\tau$ vs. $\tau/2$')]
    b.legend(handles=hs, loc='lower right', fontsize=7.5)
    panel(b, 'b', 'radius discrepancy vs. time-step effect')
    fig.savefig(F + 'fig7_droplets.pdf'); plt.close(fig)
    return dict(box=box)


def _e(v):
    if v is None:
        return '--'
    if v == 0:
        return '$0$'
    e = int(np.floor(np.log10(abs(v))))
    return r'$%.1f\cdot10^{%d}$' % (v / 10**e, e)


def table_resolution():
    """Resolution study for the two interface experiments, from summary.json (tight solver throughout)."""
    S = json.load(open(D + 'summary.json'))
    st, dr = S['star'], S['drops']
    lines = [r'\begin{tabular}{lllcccc}', r'\toprule',
             r'\multicolumn{7}{l}{\emph{Star and two circles} ($A=2$, $T=0.5$): displacement proxy $\delta_\Gamma(t)$ against $N=128$, $\tau=2.5\cdot10^{-4}$}\\',
             r'$N$ & $\tau$ & & $\delta_\Gamma(0.1)$ & $\delta_\Gamma(0.25)$ & $\delta_\Gamma(0.5)$ & $t_{m,1}$, $t_{m,2}$\\', r'\midrule']
    for key, lab in [('N128_tau0.002', ''), ('N128_tau0.001', r'$^\ast$'), ('N128_tau0.0005', ''), ('N128_tau0.00025', 'ref.'),
                     ('N256_tau0.00025', '')]:
        rr = st['runs'][key]
        lines.append(r'%d & $%g$ & %s & %s & %s & %s & $%.5f$, $%.4f$\\' % (
            rr['N'], rr['tau'], lab, _e(rr.get('delta_0.1')), _e(rr.get('delta_0.25')), _e(rr.get('delta_0.5')),
            rr['mergers'][0], rr['mergers'][1]))
    R = dr['runs']

    def dmax(a, b):
        return max(abs(x - y) for tk in R[a]['R_at'] for x, y in zip(R[a]['R_at'][tk], R[b]['R_at'][tk]))
    lines += [r'\midrule',
              r'\multicolumn{7}{l}{\emph{Unequal droplets} ($A=0$, $T=2$): radii at $t=0.15,1,2$ against the run in column 3}\\',
              r'$N$ & $\tau$ & against & \multicolumn{3}{c}{$\max_{j,k}|R_j(t_k)-R_j^{\rm cmp}(t_k)|$} & $t_1$, $t_2$\\', r'\midrule']
    rows = [('N256_tau0.004', 'N256_tau0.001'), ('N256_tau0.002', 'N256_tau0.001'), ('N256_tau0.001', None),
            ('N256_tau0.0005', 'N256_tau0.001'), ('N128_tau0.0005', 'N256_tau0.0005'), ('N512_tau0.001', 'N256_tau0.001')]
    for key, cmp_ in rows:
        if key not in R or (cmp_ and cmp_ not in R):
            continue
        rr = R[key]
        lab = r'$^\ast$' if cmp_ is None else (r'$N=%d$, $\tau=%g$' % (R[cmp_]['N'], R[cmp_]['tau']))
        lines.append(r'%d & $%g$ & %s & \multicolumn{3}{c}{%s} & $%.4f$, $%.4f$\\' % (
            rr['N'], rr['tau'], lab, '--' if cmp_ is None else _e(dmax(key, cmp_)), rr['t_dis'][0], rr['t_dis'][1]))
    lines += [r'\bottomrule', r'\end{tabular}']
    open(T + 'tab_resolution.tex', 'w').write('\n'.join(lines) + '\n')


def figS2():
    """Supplement: three seeds, mean -0.5; snapshots of the practical solver with tight contours, droplet counts,
    practical-vs-tight L2 differences with the tau/2 and 2N effects for seed 1, and transforms per step."""
    B = json.load(open(D + 'fig_seeds.json')); meta = B['meta']; L = meta['L']
    S = np.load(D + 'fig_seeds_snaps.npz')
    RF = np.load(D + 'fig_seed_refine_snaps.npz') if os.path.exists(D + 'fig_seed_refine_snaps.npz') else None
    times = ['0.5', '1', '2', '5']
    seeds = [k for k in ['seed1', 'seed2', 'seed3'] if k in B['runs']]
    N = S[seeds[0] + '_p_0'].shape[0]; xs = np.arange(N) * L / N
    H = 6.6; fig = plt.figure(figsize=(TW, H))
    w = 0.205; hh = w * TW / H; gap = 0.012
    for r_, sd in enumerate(seeds):
        y = 0.97 - (r_ + 1) * (hh + 0.022)
        for c, tk in enumerate(times):
            ax = fig.add_axes([0.04 + c * (w + gap), y, w, hh])
            p = S[f'{sd}_p_{tk}'].astype(float); t_ = S[f'{sd}_t_{tk}'].astype(float)
            im = ax.imshow(p.T, origin='lower', cmap='RdBu_r', vmin=-1.2, vmax=1.2, extent=[0, L, 0, L], interpolation='nearest', rasterized=True)
            ax.contour(xs, xs, t_.T, levels=[0.0], colors=INK, linewidths=0.35)
            ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')
            if r_ == 0:
                ax.set_title(rf'$t={tk}$', pad=3)
            if c == 0:
                ax.set_ylabel('seed %s' % sd[-1], fontsize=8.5)
    cb = fig.colorbar(im, cax=fig.add_axes([0.04 + 4 * (w + gap) + 0.004, 0.97 - 3 * (hh + 0.022) + 0.02, 0.012, 3 * hh]),
                      ticks=[-1, -0.5, 0, 0.5, 1]); cb.ax.tick_params(labelsize=7.5)
    a = fig.add_axes([0.08, 0.07, 0.25, 0.18]); b = fig.add_axes([0.44, 0.07, 0.23, 0.18]); c = fig.add_axes([0.76, 0.07, 0.22, 0.18])
    cols = {'seed1': BLUE, 'seed2': AQUA, 'seed3': ORANGE}
    for sd in seeds:
        r = B['runs'][sd]['rec']; t = np.array(r['t'])
        a.plot(t, r['ncomp_p'], '-', color=cols[sd], lw=1.0, label='seed ' + sd[-1])
        a.plot(t, r['ncomp_t'], '--', color=INK, lw=0.5)
        b.semilogy(t, r['diffL2'], '-', color=cols[sd], lw=0.9)
        tv = t[12:-12]
        c.plot(tv, np.convolve(r['fft_p'], np.ones(25) / 25, 'valid'), '-', color=cols[sd], lw=0.9)
        c.plot(tv, np.convolve(r['fft_t'], np.ones(25) / 25, 'valid'), '--', color=cols[sd], lw=0.7)
    if RF is not None:
        for key, mk, lab in [('N256_tau0.001', '*', r'$\tau$ vs. $\tau/2$'), ('N512_tau0.002', 'D', r'$N$ vs. $2N$')]:
            vals = []
            for tk in times:
                kk = f'{key}_{tk}'
                if kk in RF.files:
                    q = RF[kk].astype(float); q = q[::q.shape[0] // N, ::q.shape[0] // N]
                    vals.append((float(tk), float(np.sqrt(((q - S[f'seed1_t_{tk}'].astype(float))**2).mean()) * L)))
            if vals:
                b.semilogy(*zip(*vals), mk, color=BLUE, mec=INK, mew=0.5, ms=7 if mk == '*' else 4, label=lab + ' (seed 1, tight)')
    a.set_xlabel('$t$'); a.set_ylabel(r'components of $\{\phi>0\}$'); a.legend(fontsize=7, loc='upper right')
    panel(a, 'a', 'number of droplets')
    b.set_xlabel('$t$'); b.set_ylabel(r'$L^2$ difference'); b.set_ylim(1e-6, 1e0)
    panel(b, 'b', r'practical vs. tight (lines)')
    c.set_xlabel('$t$'); c.set_ylabel('transforms per step'); c.set_ylim(0, 50)
    c.legend(handles=[Line2D([], [], color=MUTED, label='practical'), Line2D([], [], color=MUTED, ls='--', label='tight')], fontsize=7, loc='upper right')
    panel(c, 'c', 'cost (25-step mean)')
    fig.savefig(F + 'figS2_seeds.pdf'); plt.close(fig)


def table_seeds():
    S = json.load(open(D + 'summary.json'))['seeds']
    rows = []
    for k in ['seed1', 'seed2', 'seed3']:
        v = S[k]
        rows.append(r'%s & %d / %d & %.2f & %d & %d (%d) & %s & %s & %s & %s\\' % (
            k[-1], v['fft_p'], v['fft_t'], v['ratio'], v['fails'], v['count_steps_differ'], v['count_max_diff'],
            _e(max(v['delta_at'].values())), _e(v['diffL2_max']), _e(v['TE_max']), _e(max(v['mass_drift']))))
    tab = '\n'.join([r'\begin{tabular}{lcccccccc}', r'\toprule',
                     r'seed & FFT$_{\rm p}$ / FFT$_{\rm t}$ & FFT$_{\rm t}$/FFT$_{\rm p}$ & failures & count differs (max) & $\max_k\delta_\Gamma(t_k)$ & $\max_n\|\phi_p-\phi_t\|$ & (T$_E$) ratio & mass drift\\',
                     r'\midrule'] + rows + [r'\bottomrule', r'\end{tabular}'])
    open(T + 'tab_seeds.tex', 'w').write(tab + '\n')


def table_accuracy():
    """All inner-solve variants of the accuracy test (formerly supplementary Fig. S1): observed orders between
    adjacent step sizes, error at the smallest step relative to the exact solve, mean inner iterations (range over tau)."""
    d = load_time()
    tau = np.array([r['tau'] for r in d['exact']]); ex = np.array([r['Hm1max'] for r in d['exact']])
    names = [('exact', r'Newton, $\nB{\Rn}\le10^{-12}$ (exact)'), ('w1_K1', r'$\omega=1$, $K=1$'), ('w1_K2', r'$\omega=1$, $K=2$'),
             ('w1_K6', r'$\omega=1$, $K=6$ (original code)'), ('w05_K1', r'$\omega=\frac12$, $K=1$'), ('w05_K2', r'$\omega=\frac12$, $K=2$'),
             ('w05_K4', r'$\omega=\frac12$, $K=4$'), ('w05_acc', r'$\omega=\frac12$ + (T$_A$), $c_R=1$'),
             ('armijo_acc', r'Armijo + (T$_A$), $c_R=1$'), ('armijo_acc01', r'Armijo + (T$_A$), $c_R=0.1$')]
    J = [int(round(-np.log2(t))) for t in tau]
    lines = [r'\begin{tabular}{l' + 'c' * (len(tau) - 1) + 'cc}', r'\toprule',
             r'& \multicolumn{%d}{c}{$p_j=\log_2(e_{j-1}/e_j)$, $\tau_j=2^{-j}$} & $e_{10}/e^{\rm ex}_{10}$ & mean it.\\' % (len(tau) - 1),
             r'\cmidrule(lr){2-%d}' % len(tau),
             'inner solve & ' + ' & '.join(r'$j=%d$' % j for j in J[1:]) + r' & & \\', r'\midrule']
    for k, lab in names:
        e = np.array([r['Hm1max'] for r in d[k]]); it = np.array([r['iters'] for r in d[k]])
        rates = np.log2(e[:-1] / e[1:])
        lines.append(lab + ' & ' + ' & '.join('%.2f' % r for r in rates) + r' & %.2f & %.2f--%.2f\\' % (e[-1] / ex[-1], it.min(), it.max()))
    lines += [r'\bottomrule', r'\end{tabular}']
    open(T + 'tab_accuracy.tex', 'w').write('\n'.join(lines) + '\n')


def table_summary():
    """Practical vs tight solver on all lockstep runs (from summary.json and the raw histories)."""
    S = json.load(open(D + 'summary.json'))
    B = json.load(open(D + 'fig_bench.json'))['rec']
    rows = []
    b = S['bench']
    rows.append((r'labyrinth, $512^2$, $\tau=0.004$', 2500, int(sum(B['fft_p'])), int(sum(B['fft_t'])), b['diffL2_max'], '--',
                 b['n_increase_p'] + b['n_increase_t'], b['TE_ratio_max'], max(b['mass_drift_p'], b['mass_drift_t'])))
    p_ = S['star']['prod']
    rows.append((r'merger, $128^2$, $\tau=10^{-3}$', 500, p_['fft_p'], p_['fft_t'], p_['diffL2_max'],
                 'same' if p_['mergers_p'] == p_['mergers_t'] else 'differ', sum(p_['Etil_increases']), p_['TE_max'], max(p_['mass_drift'])))
    p_ = S['drops']['prod']
    rows.append((r'ripening, $256^2$, $\tau=10^{-3}$', 2000, p_['fft_p'], p_['fft_t'], p_['diffL2_max'],
                 'same' if p_['t_dis_p'] == p_['t_dis_t'] else 'differ', sum(p_['Etil_increases']), p_['TE_max'], max(p_['mass_drift'])))
    for k in ['seed1', 'seed2', 'seed3']:
        v = S['seeds'][k]
        rows.append((r'$\bar\phi=-0.5$, seed %s, $256^2$, $\tau=0.002$' % k[-1], 2500, v['fft_p'], v['fft_t'], v['diffL2_max'],
                     '%d/2500' % v['count_steps_differ'], sum(v['Etil_increases']), v['TE_max'], max(v['mass_drift'])))
    lines = [r'\begin{tabular}{lrrrrlccc}', r'\toprule',
             r'run & steps & FFT$_{\rm p}$ & FFT$_{\rm t}$ & fewer & events & $\max_n\nrm{\phi_{\rm p}^n-\phi_{\rm t}^n}$ & (T$_E$) ratio & mass drift\\',
             r'\midrule']
    for name, st, fp, ft, dl2, ev, inc, te, md in rows:
        assert inc == 0
        lines.append(r'%s & %d & %d & %d & %.1f\%% & %s & %s & %s & %s\\' % (name, st, fp, ft, 100 * (1 - fp / ft), ev, _e(dl2), _e(te), _e(md)))
    lines += [r'\bottomrule', r'\end{tabular}']
    open(T + 'tab_summary.tex', 'w').write('\n'.join(lines) + '\n')


# ------------------------------------------------------------------ Tables
def table_space():
    rows = json.load(open(D + 'space.json'))
    lines = [r'\begin{tabular}{l' + 'c' * len(rows) + '}', r'\toprule',
             r'$N$ & ' + ' & '.join(str(r['N']) for r in rows) + r' \\', r'\midrule',
             r'$\|e\|$ at $T$ & ' + ' & '.join(('%.1e' % r['L2']).replace('e-0', 'e$-$').replace('e-', 'e$-$') for r in rows) + r' \\',
             r'\bottomrule', r'\end{tabular}']
    open(T + 'tab_space.tex', 'w').write('\n'.join(lines) + '\n')


def load_solver():
    d = json.load(open(D + 'solver_tight.json')) + json.load(open(D + 'solver_practical.json'))
    for part in ['tight', 'practical']:
        f = D + f'solver_armijo_{part}.json'
        if os.path.exists(f):
            new = json.load(open(f))
            keys = {(r['stop'], r['tau'], r['A']) for r in new}
            d = [r for r in d if not (r['method'] == 'armijo' and (r['stop'], r['tau'], r['A']) in keys)] + new
    return d


def table_solver():
    d = load_solver()
    name = {'linear_w1': r'linear, $\omega=1$', 'relax_w05': r'relaxed, $\omega=\frac12$', 'armijo': 'Armijo', 'newton': 'Newton--PCG'}
    idx = {(r['stop'], r['tau'], r['A'], r['method']): r for r in d}
    lines = [r'\begin{tabular}{ll' + 'rr' * 6 + '}', r'\toprule',
             r'& & \multicolumn{4}{c}{$\tau=0.001$} & \multicolumn{4}{c}{$\tau=0.004$} & \multicolumn{4}{c}{$\tau=0.016$}\\',
             r'\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(lr){11-14}',
             r'& & \multicolumn{2}{c}{$A=0$} & \multicolumn{2}{c}{$A=2$} & \multicolumn{2}{c}{$A=0$} & \multicolumn{2}{c}{$A=2$} & \multicolumn{2}{c}{$A=0$} & \multicolumn{2}{c}{$A=2$}\\',
             r'stop & method & it & FFT & it & FFT & it & FFT & it & FFT & it & FFT & it & FFT\\', r'\midrule']
    for stop, slab in [('tight', r'$\nB{\Rn}\le10^{-10}$'), ('practical', r'(T$_E$)+(T$_A$)')]:
        for j, m in enumerate(['linear_w1', 'relax_w05', 'armijo', 'newton']):
            cells = []
            for tau in [0.001, 0.004, 0.016]:
                for A in [0.0, 2.0]:
                    r = idx[(stop, tau, A, m)]
                    cells += [r'\multicolumn{2}{c}{div.}'] if r['status'] != 'ok' else ['%.1f' % r['mean_it'], '%.0f' % r['nfft_per_step']]
            first = (r'\multirow{4}{*}{\shortstack[l]{%s}}' % slab) if j == 0 else ''
            lines.append(first + ' & ' + name[m] + ' & ' + ' & '.join(cells) + r'\\')
        if stop == 'tight':
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    open(T + 'tab_solver.tex', 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    import sys
    os.makedirs(F, exist_ok=True)
    which = sys.argv[1:] or ['1', '2', '3', '4', '5', 'tables']
    for w in which:
        if w == 'tables':
            table_space(); table_solver(); table_accuracy()
            if os.path.exists(D + 'fig_star_prod.json'):
                table_resolution()
            if os.path.exists(D + 'fig_seeds.json'):
                table_seeds(); table_summary()
        else:
            out = globals()['fig' + w]()
            if out:
                print(out)
    print('done', which)
