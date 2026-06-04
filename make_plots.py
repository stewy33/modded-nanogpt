import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.makedirs('img', exist_ok=True)

# ---- Trajectory data (time_s, val_loss) parsed from logs/ ----
traj = {
    'baseline (bs512, 12L)':       ([60,120,180,240,300], [None]),  # placeholder, see below
}

# val_loss trajectories for key runs (from logs)
runs = {
    'baseline bs512/12L (4.1955)': ([300],[4.1955]),
    'l6_bs128 w768 (3.8041)':      ([60,120,180,240,300],[4.5019,4.1906,4.0635,3.9452,3.8041]),
    'narrow512 w512 (3.8108)':     ([60,120,180,240,300],[4.3738,4.1463,4.0380,3.9366,3.8108]),
    'BEST narrow640 w640 (3.7967)':([60,120,180,240,300],[4.4429,4.1641,4.0399,3.9298,3.7967]),
    'l7_w640 (3.7983)':            ([60,120,180,240,300],[4.4600,4.1786,4.0508,3.9370,3.7983]),
}

plt.figure(figsize=(8,5))
for name,(t,v) in runs.items():
    if len(t)==1:
        plt.scatter(t,v,marker='x',s=80,label=name)
    else:
        lw = 2.6 if name.startswith('BEST') else 1.4
        plt.plot(t,v,marker='o',ms=3,lw=lw,label=name)
plt.xlabel('training time (s)'); plt.ylabel('val_loss')
plt.title('Validation loss trajectories (5-min single-GPU budget)')
plt.ylim(3.7,4.7); plt.grid(alpha=0.3); plt.legend(fontsize=8)
plt.tight_layout(); plt.savefig('img/trajectories.png', dpi=120); plt.close()

# ---- Width sweep (the session-2 lever) ----
widths = [512,640,768,1024]
wloss  = [3.8108,3.7967,3.8041,None]  # 1024 did not finish (lost hard)
plt.figure(figsize=(6,4))
xs=[w for w,l in zip(widths,wloss) if l]; ys=[l for l in wloss if l]
plt.plot(xs,ys,'o-',color='C2')
for w,l in zip(xs,ys):
    plt.annotate(f'{l:.4f}',(w,l),textcoords='offset points',xytext=(0,7),ha='center',fontsize=9)
plt.scatter([1024],[3.815],marker='x',color='red',s=90)
plt.annotate('1024: lost\n(245ms/step)',(1024,3.815),textcoords='offset points',xytext=(-5,-28),ha='center',fontsize=8,color='red')
plt.axhline(3.8041,ls='--',color='gray',alpha=0.6,label='prev best (w768) 3.8041')
plt.xlabel('n_embd (width), 6 layers, bs128'); plt.ylabel('final val_loss')
plt.title('Width sweep — optimum at n_embd=640')
plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig('img/width_sweep.png', dpi=120); plt.close()

# ---- Final bar of all completed configs ----
configs = [
    ('baseline\nbs512/12L',4.1955),
    ('bs128\nw768/12L',3.8619),
    ('l6_bs128\nw768',3.8041),
    ('narrow512\nw512',3.8108),
    ('l7_w640',3.7983),
    ('wd40\nw640',3.7971),
    ('BEST\nnarrow640',3.7967),
]
configs.sort(key=lambda x:-x[1])
names=[c[0] for c in configs]; vals=[c[1] for c in configs]
colors=['C3' if 'BEST' in n else 'C0' for n in names]
plt.figure(figsize=(8,4.5))
b=plt.bar(names,vals,color=colors)
plt.ylim(3.7,4.25); plt.ylabel('final val_loss')
plt.title('Configs vs final val_loss (lower=better)')
for r,v in zip(b,vals):
    plt.annotate(f'{v:.4f}',(r.get_x()+r.get_width()/2,v),textcoords='offset points',xytext=(0,3),ha='center',fontsize=8)
plt.xticks(fontsize=7); plt.grid(axis='y',alpha=0.3); plt.tight_layout()
plt.savefig('img/final_bar.png', dpi=120); plt.close()
print('plots written')
