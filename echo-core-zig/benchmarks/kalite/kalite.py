import sys, numpy as np
O, ad = sys.argv[1], sys.argv[2]
tok = [int(x) for x in open(O+'/tok.txt').read().split(',')]
def load(a):
    x = np.fromfile(f'{O}/logits_{a}.bin', dtype=np.float32); return x.reshape(768, -1)
L = load(ad); y = np.array(tok[1:769])
m = L.max(1, keepdims=True); lse = (m[:,0] + np.log(np.exp(L-m).sum(1)))
nll = lse - L[np.arange(768), y]
top = L.argmax(1)
s = f'{ad}: ppl {np.exp(nll.mean()):.3f}  top1-dogru {np.mean(top==y)*100:.1f}%'
try:
    B = load('base'); s += f'  base-top1-uyum {np.mean(top==B.argmax(1))*100:.1f}%'
except Exception: pass
print(s)
