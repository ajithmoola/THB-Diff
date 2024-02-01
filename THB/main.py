from THB.funcs import *

p = 2
kv = np.array([0, 0, 0, 0.2, 0.4, 0.6, 0.8, 1, 1, 1])
num_uknots = len(np.unique(kv))
num_fns = num_uknots + p
num_cp = num_fns

# param = np.linspace(0, 1, 100).tolist()
# print
# spans = [findSpan(num_cp-1, p, u, kv) for u in param]
# fns = [basisFun(spans[i], param[i], p, kv) for i in range(100)]
span = findSpan(num_cp-1, p, 0, kv)
fns = basisFun(span, 0, p, kv)
print(span, fns)