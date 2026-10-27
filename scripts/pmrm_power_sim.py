"""Reference power simulation for the stage 0 gate (spec v1.5, section 6).

Illustrative only. It does NOT simulate the planned primary comparison; that must be simulated directly.

Assumptions:
- One conditioning variable x and one portfolio P&L y, standardized; true correlation rho.
- Rows independent (no serial dependence); Gaussian, or Student-t with 3 df scaled to unit variance.
- Rolling training window of 120 rows, refit every 20 rows, sample moments (no shrinkage).
- M0: unconditional Gaussian for y. M1: Gaussian conditional on x from the fitted bivariate moments.
- Score: per-row Gaussian negative log score; statistic = mean of (M0 - M1) paired differences.
- Test: one-sided z-test, mean / standard error > 1.96, iid standard error (no block bootstrap).
- 1,000 replications per cell, seed 1729.
Output: P(M1 beats M0 on average) and P(significant) per (rho, test rows).
"""
# Power of v1.2's primary test (M1 conditional vs M0 unconditional, Gaussian NLS)
# under the spec's own sizes: rolling train 120, refit every 20, test N rows.
import numpy as np
rng=np.random.default_rng(1729)
def nls(y,m,v): return 0.5*(np.log(2*np.pi*v)+(y-m)**2/v)
def run(rho,ntest,ntrain=120,refit=20,reps=2000,df=None):
    wins=0; sig=0; worse=0
    for _ in range(reps):
        n=ntrain+ntest
        if df is None:
            x=rng.standard_normal(n); e=rng.standard_normal(n)
        else:  # fat tails, student-t scaled to unit var
            x=rng.standard_t(df,n)/np.sqrt(df/(df-2)); e=rng.standard_t(df,n)/np.sqrt(df/(df-2))
        y=rho*x+np.sqrt(1-rho**2)*e
        d=[]
        for s in range(ntrain,n,refit):
            X=x[s-ntrain:s]; Y=y[s-ntrain:s]
            mx,my=X.mean(),Y.mean(); vx,vy=X.var(),Y.var(); c=((X-mx)*(Y-my)).mean()
            b=c/vx; vc=vy-c*c/vx
            for t in range(s,min(s+refit,n)):
                l0=nls(y[t],my,vy); l1=nls(y[t],my+b*(x[t]-mx),vc)
                d.append(l0-l1)   # >0 means M1 better
        d=np.array(d); m=d.mean(); se=d.std(ddof=1)/np.sqrt(len(d))
        wins+=m>0; sig+=m/se>1.96; worse+=m<0
    return wins/reps, sig/reps
print("rho  ntest  P(M1 beats M0)  P(significant at 5%)   [Gaussian]")
for rho in [0.05,0.10,0.15,0.20,0.30]:
    for nt in [40,200,1000]:
        w,s=run(rho,nt,reps=1000)
        print(f"{rho:.2f} {nt:6d}   {w:.2f}            {s:.2f}")
print("\nfat tails (t, df=3), rho=0.10 / 0.20:")
for rho in [0.10,0.20]:
    for nt in [40,200,1000]:
        w,s=run(rho,nt,reps=1000,df=3)
        print(f"{rho:.2f} {nt:6d}   {w:.2f}            {s:.2f}")
