# CAGE Filament UNet - public copy of our own code (current v2 code, 1024 px; v1 code not retrievable). Our entry: PQ 0.33.
# CAGE filament seg: trains own U-Net (smp, resnet34 imagenet enc) on MAGFiLO train, PQ-tuned postproc
import subprocess,sys
subprocess.run([sys.executable,'-m','pip','install','-q','segmentation-models-pytorch','pycocotools'])
import json,os,time,numpy as np,cv2,torch,torch.nn.functional as F,pandas as pd
import segmentation_models_pytorch as smp
from pycocotools import mask as mu
T0=time.time()
R='/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/'
if not os.path.exists(R):
    import glob; R=glob.glob('/kaggle/input/**/MAGFiLO_1.0_Kaggle_2026/',recursive=True)[0]
SMOKE=int(os.environ.get("SMOKE","1"))  # public CPU copy: smoke run; full training used SMOKE=0 on a T4 GPU
S=1024; dev='cuda' if torch.cuda.is_available() else 'cpu'; print(dev,R)
d=json.load(open(R+'train/MAGFiLO_1.0_Annotations_kaggle2026_train.json'))
files=set(os.listdir(R+'train/train_images'))
imgs={i['id']:i for i in d['images'] if i['file_name'] in files}
anns={}
for a in d['annotations']:
    if a['image_id'] in imgs: anns.setdefault(a['image_id'],[]).append(a)
def decode(seg):
    if isinstance(seg,list): rle=mu.merge(mu.frPyObjects(seg,2048,2048))
    elif isinstance(seg['counts'],list): rle=mu.frPyObjects(seg,2048,2048)
    else: rle=seg
    return mu.decode(rle).astype(bool)
ids=sorted(imgs)[:40] if SMOKE else sorted(imgs); X=[];Y=[]
for iid in ids:
    g=cv2.imread(R+'train/train_images/'+imgs[iid]['file_name'],0)
    m=np.zeros((2048,2048),np.uint8)
    for a in anns.get(iid,[]): m[decode(a['segmentation'])]=1
    X.append(cv2.resize(g,(S,S),interpolation=cv2.INTER_AREA)); Y.append(cv2.resize(m*255,(S,S),interpolation=cv2.INTER_AREA))
X=np.array(X);Y=(np.array(Y)>127).astype(np.uint8); print('loaded',X.shape,Y.mean(),time.time()-T0)
rng=np.random.RandomState(0); perm=rng.permutation(len(ids)); nv=10 if SMOKE else 100; vi,ti=perm[:nv],perm[nv:]
def norm(x): return (torch.tensor(x,dtype=torch.float32)/255.-0.5)/0.25
model=smp.Unet('resnet34',encoder_weights='imagenet',in_channels=1,classes=1).to(dev)
dice=smp.losses.DiceLoss('binary'); bce=torch.nn.BCEWithLogitsLoss()
EP=1 if SMOKE else 34; BS=4
opt=torch.optim.AdamW(model.parameters(),3e-4,weight_decay=1e-4)
sch=torch.optim.lr_scheduler.OneCycleLR(opt,1e-3,total_steps=EP*(len(ti)//BS+1))
scaler=torch.cuda.amp.GradScaler(enabled=dev=='cuda')
def predict(xs):
    model.eval(); out=[]
    with torch.no_grad(), torch.autocast(dev,enabled=dev=='cuda'):
        for i in range(0,len(xs),8):
            x=norm(xs[i:i+8])[:,None].to(dev)
            p=torch.sigmoid(model(x))+torch.sigmoid(model(x.flip(-1))).flip(-1)
            out.append((p/2)[:,0].float().cpu().numpy())
    return np.concatenate(out)
for ep in range(EP):
    model.train(); rng.shuffle(ti); tl=0
    for i in range(0,len(ti),BS):
        b=ti[i:i+BS]; x=norm(X[b])[:,None]; y=torch.tensor(Y[b]).float()[:,None]
        if rng.rand()<.5: x,y=x.flip(-1),y.flip(-1)
        if rng.rand()<.5: x,y=x.flip(-2),y.flip(-2)
        x=x*(0.8+0.4*rng.rand())+0.2*(rng.rand()-.5)
        x,y=x.to(dev),y.to(dev)
        with torch.autocast(dev,enabled=dev=='cuda'):
            lo=model(x); loss=dice(lo,y)+bce(lo,y)
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); tl+=loss.item()
    print(ep,round(tl,3),round(time.time()-T0))
# instances + PQ
def instances(p,th,minA,close):
    m=(cv2.resize(p,(2048,2048))>th).astype(np.uint8)
    lm=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(close,close))) if close>1 else m
    n,lab,st,_=cv2.connectedComponentsWithStats(lm,8)
    return [ (lab==k)&(m>0) for k in range(1,n) if st[k,4]>=minA ]
def pq(gts,prs):
    tp=fp=fn=0;s=0
    for g,pr in zip(gts,prs):
        used=set()
        for gm in g:
            best=0;bj=-1
            for j,pm in enumerate(pr):
                if j in used: continue
                inter=(gm&pm).sum()
                if inter==0: continue
                iou=inter/(gm|pm).sum()
                if iou>best: best,bj=iou,j
            if best>0.5: tp+=1;s+=best;used.add(bj)
            else: fn+=1
        fp+=len(pr)-len(used)
    return s/max(tp+.5*fp+.5*fn,1e-9)
vp=predict(X[vi]); gts=[[decode(a['segmentation']) for a in anns.get(ids[k],[])] for k in vi]
best=(-1,None)
for th in [.4,.5,.6]:
  for minA in [100,250,500]:
    for close in [15,31,51]:
      s=pq(gts,[instances(p,th,minA,close) for p in vp]); print('PQ',th,minA,close,round(s,4))
      if s>best[0]: best=(s,(th,minA,close))
print('BEST val PQ',best,round(time.time()-T0))
th,minA,close=best[1]
tf=sorted(os.listdir(R+'test/test_images'))[:5 if SMOKE else None]
TX=np.array([cv2.resize(cv2.imread(R+'test/test_images/'+f,0),(S,S),interpolation=cv2.INTER_AREA) for f in tf])
tp_=predict(TX); rows=[]
for f,p in zip(tf,tp_):
    for k,m in enumerate(instances(p,th,minA,close),1):
        r=mu.encode(np.asfortranarray(m.astype(np.uint8))); rows.append((f[:-5]+f'_{k}',r['counts'].decode()))
pd.DataFrame(rows,columns=['filament_id','segmentation_rle']).to_csv('submission.csv',index=False)
print('wrote',len(rows),'rows for',len(tf),'images',round(time.time()-T0))
