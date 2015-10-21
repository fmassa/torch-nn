local SpatialConvolution, parent =
                  torch.class('nn.SpatialConvolutionGroups', 'nn.Module')

function SpatialConvolution:__init(nInputPlane, nOutputPlane,
                                   kW, kH, dW, dH, padW, padH, groups)
  parent.__init(self)
  assert(nInputPlane % groups == 0,
  'nInputPlane should be divisible by nGroups')
  assert(nOutputPlane % groups == 0,
  'nOutputPlane should be divisible by nGroups')

  self.modules = nn.Concat(2)

  for i=1,groups do
    local n = nn.Sequential()
    n:add(nn.Narrow(2,(i-1)*nInputPlane/groups+1,nInputPlane/groups))
    n:add(nn.SpatialConvolution(nInputPlane/groups,nOutputPlane/groups,
    kW, kH, dW,dH,padW,padH))
    self.modules:add(n)
  end
end

function SpatialConvolution:updateOutput(input)
  self.output = self.modules:updateOutput(input)
  return self.output
end

function SpatialConvolution:updateGradInput(input, gradOutput)
  self.gradInput = self.modules:updateGradInput(input, gradOutput)
  return self.gradInput
end

function SpatialConvolution:accGradParameters(input, gradOutput, scale)
  return self.modules:accGradParameters(input, gradOutput, scale)
end

function SpatialConvolution:type(type, tensorCache)
  self.modules:type(type, tensorCache)
  return self
end

function SpatialConvolution:parameters()
  return self.modules:parameters()
end


