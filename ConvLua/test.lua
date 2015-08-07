require 'nn'
dofile 'ConvLua.lua'

k = 5
d = 1
p = 1
iter = 10

m = nnf.SpatialConvolution(3,96,k,k,d,d,p,p)
mm = nn.SpatialConvolutionMM(3,96,k,k,d,d,p,p)
a = torch.rand(32,3,100,100)

--m = nnf.SpatialConvolution(2,2,3,3,1,1,1,1)
--a = torch.rand(2,2,3,3)

print('Warming up')
for i=1,3 do
  o = m:forward(a)
  o2 = m:updateOutput2(a)
  o3 = m:updateOutput3(a)
  o22 = m:updateOutput4(a)
  o4 = mm:forward(a)
end

timer = torch.Timer()

print('T1')
timer:reset()
for i=1,iter do
  o = m:forward(a)
end
t1 = timer:time().real
print('T2')
timer:reset()
for i=1,iter do
  o2 = m:updateOutput2(a)
end
t2 = timer:time().real
print('T3')
timer:reset()
for i=1,iter do
  o3 = m:updateOutput3(a)
end
t3 = timer:time().real
print('T4')
timer:reset()
for i=1,iter do
  o22 = m:updateOutput4(a)
end
t4 = timer:time().real
print('T5')
timer:reset()
for i=1,iter do
  o4 = mm:forward(a)
end
t5 = timer:time().real
timer:stop()

tt = {['T1_Caffe_kern']  = t1,
      ['T2_Torch_kern']  = t2,
      ['T3_Caffe_par']   = t3,
      ['T4_Torch_fill']  = t4,
      ['T5_Torch_NN']    = t5}
print('Timings:')
print(tt)
