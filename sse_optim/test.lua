dofile 'sse.lua'

budget = 1e7*100
n = 1e3
iter = budget/n

a = torch.ones(n)
print('Warming up')
for i=1,10 do
  sseMul(a,1.0)
  Mul(a,1.0)
  parMul(a,1.0)
  parsseMul(a,1.0)
  a:mul(1.0)
end

print('T1')
timer = torch.Timer()
timer:reset()
for i=1,iter do
  sseMul(a,1.01)
end
t1 = timer:time().real
a = torch.ones(n)
collectgarbage()
print('T2')
timer:reset()
for i=1,iter do
  Mul(a,1.01)
end
t2 = timer:time().real
a = torch.ones(n)
collectgarbage()
print('T3')
timer:reset()
for i=1,iter do
  parMul(a,1.01)
end
t3 = timer:time().real
a = torch.ones(n)
collectgarbage()
print('T4')
timer:reset()
for i=1,iter do
  parsseMul(a,1.01)
end
t4 = timer:time().real
a = torch.ones(n)
collectgarbage()
print('T5')
timer:reset()
for i=1,iter do
  a:mul(1.01)
end
t5 = timer:time().real


tt = {['sseMul']      = t1,
      ['Mul']         = t2,
      ['parMul']      = t3,
      ['parsseMul']   = t4,
      ['torch.mul']   = t5
     }

print(tt)
