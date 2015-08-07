Small benchmark on SSE/OMP optimizations
========================================

Setup: Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz

## Multiplication by scalar
####Using 12 threads

For the default parameters (`-n 2e7 -budget 1e9`), I get:
```lua
{
  budget : 1000000000
  n      : 20000000
}
Warming up
T1
T2
T3
T4
T5
{
  sseMul    : 0.79944682121277
  Mul       : 0.72756409645081
  parMul    : 0.38451194763184
  parsseMul : 0.39017486572266
  torch.mul : 0.40270113945007
}
```

For `-n 2e8 -budget 1e9`

```lua
{
  budget : 1000000000
  n      : 200000000
}
Warming up
T1
T2
T3
T4
T5
{
  sseMul    : 0.75903820991516
  Mul       : 0.71702790260315
  parMul    : 0.36014199256897
  parsseMul : 0.35635113716125
  torch.mul : 0.36806893348694
}
```
