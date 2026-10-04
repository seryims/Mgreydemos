import numpy as np, json
from scipy.signal import butter, sosfilt
from scipy.io import wavfile
SR=48000; DUR=32.0; N=int(SR*DUR)
rng=np.random.default_rng(3)
t=np.arange(N)/SR
def lp(x,f,o=2): return sosfilt(butter(o,f,'low',fs=SR,output='sos'),x)
def hp(x,f,o=2): return sosfilt(butter(o,f,'high',fs=SR,output='sos'),x)
def bp(x,a,b,o=2): return sosfilt(butter(o,[a,b],'band',fs=SR,output='sos'),x)
def env(n,a,d): 
    e=np.ones(n); ai=int(a*SR); e[:ai]=np.linspace(0,1,ai) if ai>0 else 1
    return e*np.exp(-np.arange(n)/SR/d)
def place(buf,x,at,g=1.0,pan=0.0):
    i=int(at*SR); x=x[:max(0,len(buf[0])-i)]
    buf[0][i:i+len(x)]+=x*g*np.sqrt(1-pan)/np.sqrt(1) ; buf[1][i:i+len(x)]+=x*g*np.sqrt(1+pan)/np.sqrt(1)
def m2f(m): return 440*2**((m-69)/12)
def saw(f,tt): return 2*((tt*f)%1)-1

music=[np.zeros(N),np.zeros(N)]; sfx=[np.zeros(N),np.zeros(N)]
BEAT=60/100; BAR=4*BEAT
prog=[[53,57,60,64],[55,59,62,67],[52,55,59,62],[57,60,64,67]]  # Fmaj7 G Em7 Am7
# expiry minor chord region handled by filter
# --- pad
pad=np.zeros(N)
nb=int(DUR/BAR)+1
for b in range(nb):
    ch=prog[b%4]; s0=b*BAR; n=int(BAR*SR)+int(.6*SR)
    tt=np.arange(n)/SR; x=np.zeros(n)
    for m in ch:
        for d in (-0.08,0.08): x+=saw(m2f(m+d/1),tt)*0.5
    e=np.minimum(1,tt/0.5)*np.minimum(1,np.maximum(0,(BAR+0.6-tt)/0.6))
    i=int(s0*SR); seg=x*e; L=min(n,N-i)
    if L>0: pad[i:i+L]+=seg[:L]
pad=lp(pad,1400,2)*0.05
# --- arp pluck (from 2.6s)
arp=np.zeros(N)
step=BEAT/2
for j in range(int(DUR/step)):
    st=j*step
    if st<2.6 or 19.8<st<24.3 or st>30: continue
    ch=prog[int(st/BAR)%4]; m=ch[[0,2,1,3,2,1,3,2][j%8]]+12
    n=int(.35*SR); tt=np.arange(n)/SR
    x=(np.sin(2*np.pi*m2f(m)*tt)+0.3*np.sin(4*np.pi*m2f(m)*tt))*np.exp(-tt/0.09)
    i=int(st*SR); L=min(n,N-i); arp[i:i+L]+=x[:L]*0.055
# --- bass + kick + hats
bass=np.zeros(N); kick=np.zeros(N); hat=np.zeros(N); duck=np.ones(N)
for j in range(int(DUR/BEAT)+1):
    st=j*BEAT
    if st<2.55 or st>30.6: continue
    expir=19.8<st<24.3
    ch=prog[int(st/BAR)%4]
    n=int(BEAT*SR); tt=np.arange(n)/SR
    f=m2f(ch[0]-24); bx=np.sin(2*np.pi*f*tt)*np.minimum(1,tt/0.01)*np.exp(-tt/0.35)
    i=int(st*SR); L=min(n,N-i); bass[i:i+L]+=bx[:L]*(0.12 if expir else 0.2)
    if not expir:
        kn=int(.4*SR); kt=np.arange(kn)/SR; ph=2*np.pi*np.cumsum(45+110*np.exp(-kt/0.04))/SR
        kx=np.sin(ph)*np.exp(-kt/0.16); L=min(kn,N-i); kick[i:i+L]+=kx[:L]*0.32
        dl=int(.3*SR); L=min(dl,N-i); duck[i:i+L]=np.minimum(duck[i:i+L],0.45+0.55*np.linspace(0,1,dl)[:L]**0.6)
        if st>7.3:
            hn=int(.08*SR); hx=lp(hp(rng.standard_normal(hn),5000),9000)*np.minimum(1,np.arange(hn)/SR/0.004)*np.exp(-np.arange(hn)/SR/0.02)
            i2=int((st+BEAT/2)*SR); L=min(hn,N-i2)
            if L>0: hat[i2:i2+L]+=hx[:L]*0.022
mus=(pad+arp)*duck+bass*duck**0.5+kick+hat
# expiry: lowpass section + global volume automation
lpmix=np.clip((t-19.6)/0.4,0,1)*np.clip((24.3-t)/0.15,0,1)
mus=mus*(1-lpmix)+lp(mus,350,2)*lpmix*1.3
vol=np.clip(t/1.0,0,1)*np.clip((32-t)/1.6,0,1)
mus*=vol
for c in (0,1): music[c]=mus.copy()
# stereo widening of pad
music[0]+=lp(np.roll(pad,240),1000)*0.4*vol; music[1]+=lp(np.roll(pad,-240),1000)*0.4*vol

# ---------- SFX ----------
def noise(n): return rng.standard_normal(n)
def whoosh(d=.6,lo=300,hi=4000,g=.35):
    n=int(d*SR); x=noise(n); tt=np.arange(n)/SR
    out=np.zeros(n); seg=int(SR*.02)
    for s in range(0,n,seg):
        p=s/n; fc=lo*(hi/lo)**(np.sin(np.pi*p)); 
        out[s:s+seg]=bp(x[max(0,s-2000):s+seg],fc*0.7,min(fc*1.4,20000))[-len(x[s:s+seg]):]
    e=np.sin(np.pi*np.clip(tt/d,0,1))**2
    return out*e*g
def tap():
    n=int(.05*SR); tt=np.arange(n)/SR
    return (np.sin(2*np.pi*1800*tt)*np.exp(-tt/0.006)*0.3+hp(noise(n),2000)*np.exp(-tt/0.004)*0.25)
def pop(f0=900,f1=520,g=.3):
    n=int(.12*SR); tt=np.arange(n)/SR; f=f1+(f0-f1)*np.exp(-tt/0.02)
    return np.sin(2*np.pi*np.cumsum(f)/SR)*np.exp(-tt/0.04)*g
def bell(f,d=.8,g=.2):
    n=int(d*SR); tt=np.arange(n)/SR
    return (np.sin(2*np.pi*f*tt)+.4*np.sin(2*np.pi*f*2.76*tt)*np.exp(-tt/0.1)+.2*np.sin(2*np.pi*f*5.4*tt)*np.exp(-tt/0.05))*np.exp(-tt/(d/3))*g
def impact(g=.6):
    n=int(1.4*SR); tt=np.arange(n)/SR; ph=2*np.pi*np.cumsum(38+80*np.exp(-tt/0.08))/SR
    return (np.sin(ph)*np.exp(-tt/0.45)*0.8+lp(noise(n),900)*np.exp(-tt/0.12)*0.5)*g
def riser(d=1.7,g=.25):
    n=int(d*SR); tt=np.arange(n)/SR; x=hp(noise(n),1500)*(tt/d)**2
    f=200*np.exp(np.log(8)*tt/d); s=np.sin(2*np.pi*np.cumsum(f)/SR)*(tt/d)**2*.4
    return (x*.5+s)*g
def swell(d=2.5,g=.25):
    n=int(d*SR); tt=np.arange(n)/SR; x=lp(noise(n),600)*np.sin(np.pi*tt/d)**2
    return x*g
def soft_whoosh(d=.9,g=.08):
    n=int(d*SR); tt=np.arange(n)/SR
    x=lp(bp(noise(n),200,1400),1200)
    e=np.sin(np.pi*tt/d)**3
    return x*e*g
def soft_bell(f,d=1.6,g=.05):
    n=int(d*SR); tt=np.arange(n)/SR
    x=np.sin(2*np.pi*f*tt)+.15*np.sin(4*np.pi*f*tt)
    e=np.minimum(1,tt/0.035)*np.exp(-tt/(d/3.5))
    return lp(x*e,2500)*g
def soft_boom(g=.25):
    n=int(1.8*SR); tt=np.arange(n)/SR
    x=np.sin(2*np.pi*55*tt)*np.minimum(1,tt/0.05)*np.exp(-tt/0.6)
    return x*g
cues=json.load(open('cues.json'))
for c in cues:
    s=c['s']; at=c['t']
    if s in ('whoosh','sheet'): place(sfx,soft_whoosh(.9,.07),at-.45,1)
    elif s=='swell': place(sfx,swell(2.5,.25),at,1)
    elif s=='impact': place(sfx,soft_boom(.22),at-.03,1)
    elif s=='success':
        for j,f in enumerate([523.25,659.25,783.99]): place(sfx,soft_bell(f,1.8,.045),at+j*0.09,1,(j-1)*.3)
    elif s=='success_short':
        for j,f in enumerate([659.25,783.99]): place(sfx,soft_bell(f,1.4,.04),at+j*0.08,1)
    elif s=='notif': place(sfx,soft_bell(880,1.4,.04),at,1); place(sfx,soft_bell(1046.5,1.6,.035),at+.15,1)
    elif s=='down': place(sfx,soft_bell(220,2.0,.08),at,1)
    elif s=='riser': place(sfx,lp(riser(1.7,.12),2000),at,1)
    # taps, pops, clicks, typing, confetti, chips, lock clicks: removed
# reverb-ish: sum of delays on sfx
for c in (0,1):
    x=sfx[c]; r=np.zeros(N)
    for d,g in [(0.031,.3),(0.047,.25),(0.071,.2),(0.113,.15),(0.163,.1),(0.241,.07)]:
        i=int((d+(0.004 if c else 0))*SR); r[i:]+=x[:-i]*g
    sfx[c]=x+lp(r,4000)*0.6
out=np.stack([music[0]+sfx[0],music[1]+sfx[1]],1)
out=np.tanh(out*1.4)/1.4
out/=np.max(np.abs(out))*1.12
wavfile.write('audio.wav',SR,(out*32767).astype(np.int16))
print('ok',out.shape)
