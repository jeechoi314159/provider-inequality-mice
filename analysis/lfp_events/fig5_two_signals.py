# Fig. 5c–e draft: two withdrawal-linked prefrontal signals (reference pipeline, cohort A non-providers)
import pickle, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scan import resid_mat
S1='#2a78d6'; S2='#eb6834'; INK='#0b0b0b'; INK2='#52514e'; GRID='#e4e3df'; SURF='#fcfcfb'
plt.rcParams.update({'font.size':8,'axes.edgecolor':INK2,'axes.labelcolor':INK,'xtick.color':INK2,'ytick.color':INK2,'axes.linewidth':0.6})
d=pickle.load(open('ext_data_s.pkl','rb')); T=d['T']
w=pickle.load(open('ext_s_wait_NP.pkl','rb')); f=pickle.load(open('ext_s_follow_NP.pkl','rb')); cl=pd.read_csv('ext_clusters_smoothed.csv')
m=pd.read_csv('two_signals_trials.csv'); m['logL']=np.log10(m.L.where(m.L>0)); NP=['A1','A3','A4','A5','A6']
fig,ax=plt.subplots(1,3,figsize=(10.8,3.2),gridspec_kw=dict(width_ratios=[1.2,1.2,1.1]),facecolor=SURF)
panels=[('PFC','g51_100','c','Before the act: PFC γ (51–100 Hz)',(-2,0)),('PFC','hbeta','d','After the act: PFC high β (24–32 Hz)',(0.25,2.25))]
for a,(reg,band,letter,title,win) in zip(ax[:2],panels):
    ki=w['keys'].index((reg,band))
    a.axvspan(*win,color=GRID,alpha=0.6,lw=0)
    a.axhline(0,color=INK2,lw=0.6); a.axvline(0,color=INK2,lw=0.6,ls='--'); a.axhline(w['thr'],color=INK2,lw=0.6,ls=':')
    a.plot(T,-w['T0'][ki],color=S1,lw=2,label='Shorter wait (followers)'); a.plot(T,f['T0'][ki],color=S2,lw=2,label='Follows vs stays out')
    for name,col,yb in [('wait_NP',S1,-1.85),('follow_NP',S2,-2.05)]:
        for _,r in cl[(cl.analysis==name)&(cl.region==reg)&(cl.band==band)&(cl.P_family<0.05)].iterrows():
            a.plot([r.t_start,r.t_end],[yb,yb],color=col,lw=3,solid_capstyle='butt')
    if band=='g51_100':   # discovery cluster in the reference slices (2-D scan, 51–100 Hz, −2.0…+0.625 s, family-wise P = 0.048)
        a.plot([-2.0,0.625],[-1.85,-1.85],color=S1,lw=3,solid_capstyle='butt')
    a.set_xlim(-6,3); a.set_ylim(-2.3,4.3); a.set_title(title,color=INK,fontsize=8.5,loc='left')
    a.set_xlabel("Time from cagemate's entry (s)"); a.spines[['top','right']].set_visible(False); a.set_facecolor(SURF)
    a.text(-0.12,1.08,letter,transform=a.transAxes,fontweight='bold',va='top')
ax[0].set_ylabel('Within-mouse t (session trend removed)'); ax[0].legend(frameon=False,loc='upper left',fontsize=7,labelcolor=INK)
# e: joint model, partial r per mouse (oriented: + = more engagement)
a=ax[2]; rows=[('logL','S_pre','Wait ← before-act γ',S1,-1),('logL','S_post','Wait ← after-act β',S1,-1),('follow','S_pre','Follow ← before-act γ',S2,1),('follow','S_post','Follow ← after-act β',S2,1)]
Pj={'logL':{'S_pre':0.0009,'S_post':0.13},'follow':{'S_pre':0.38,'S_post':0.028}}
def partial(S,ycol,sig):
    other='S_post' if sig=='S_pre' else 'S_pre'; g=S.Mouse_ID.values; pos=S.pos.values
    Z=resid_mat(S[[sig,other,ycol]].values.astype(float),g,pos)
    ex=lambda v: v-Z[:,[1]]@np.linalg.lstsq(Z[:,[1]],v,rcond=None)[0]
    return np.corrcoef(ex(Z[:,0]),ex(Z[:,2]))[0,1]
for i,(ycol,sig,lab,col,sign) in enumerate(rows):
    base=m[m.Mouse_ID.isin(NP)].dropna(subset=['S_pre','S_post'])
    S=base[(base.follow==1)&(base.L>=3)] if ycol=='logL' else base
    S=S.reset_index(drop=True); pooled=sign*partial(S,ycol,sig)
    per=[sign*partial(S[S.Mouse_ID==u].reset_index(drop=True),ycol,sig) for u in NP]
    a.plot(per,[i]*5,'o',ms=5,mfc=col,mec=SURF,mew=1,alpha=0.55)
    a.plot([pooled],[i],'D',ms=8,mfc=col,mec=INK,mew=0.8)
    a.text(1.02,1-(i+0.5)/4,f'P = {Pj[ycol][sig]:g}',transform=a.transAxes,va='center',ha='left',color=INK2,fontsize=7)
a.axvline(0,color=INK2,lw=0.6); a.set_yticks(range(4)); a.set_yticklabels([r[2] for r in rows]); a.set_xlim(-0.2,0.45); a.set_ylim(3.5,-0.5)
a.set_xlabel('Within-mouse partial r (+ = engagement)'); a.set_title('Both signals in one model',color=INK,fontsize=8.5,loc='left')
a.spines[['top','right']].set_visible(False); a.set_facecolor(SURF); a.text(-0.62,1.08,'e',transform=a.transAxes,fontweight='bold',va='top')
a.plot([],[],'D',ms=6,mfc=INK2,mec=INK,label='All trials'); a.plot([],[],'o',ms=5,mfc=INK2,mec=SURF,alpha=0.55,label='Each mouse'); a.legend(frameon=False,fontsize=7,loc='upper center',bbox_to_anchor=(0.5,-0.2),ncol=2,labelcolor=INK)
fig.tight_layout(); fig.savefig('Fig5cde_two_signals_draft_20260922.png',dpi=300,facecolor=SURF)
