Small benchmark on SSE/OMP optimizations
========================================

Setup: Intel(R) Core(TM) i7-5820K CPU @ 3.30GHz

## Multiplication by scalar

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
    Mul       : 0.70771408081055
    parsseMul : 0.39401412010193
    torch.mul : 0.37654089927673
    parMul    : 0.39146304130554
    sseMul    : 0.81081604957581
}
```

For `-n 2e7 -budget 1e9`

```lua
{
  budget : 1000000000
  n      : 100000000
}
Warming up
T1
T2
T3
T4
T5
{
  Mul       : 0.71802997589111
  parsseMul : 0.35294795036316
  torch.mul : 0.37060213088989
  parMul    : 0.36420106887817
  sseMul    : 0.74729180335999
}

```
