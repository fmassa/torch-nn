local ffi = require 'ffi'

local SpatialAdaptivePooling, parent = torch.class('nn.SpatialAdaptivePooling', 'nn.Module')

ffi.cdef[[
void SpatialAdaptiveMaxPooling_updateOutput(THCudaTensor* input, THCudaTensor* output, THCudaTensor* indices, int kW, int kH);
]]

local C = ffi.load './build/libadapt.so'

function SpatialAdaptivePooling:__init(W, H)
  parent.__init(self)
  
  self.W = W
  self.H = H

  self.output = torch.Tensor()
  
  self.indices = torch.Tensor()
  self:cuda()
end

function SpatialAdaptivePooling:updateOutput(input)
  -- its always in batch mode
  --local batchMode = input:nDimension() == 4 and true or false
  
  local dimw = 3
  local dimh = 2
  if input:nDimension() == 4 then 
    dimw = dimw + 1
    dimh = dimh + 1
  end
  
  assert(input:size(dimw)>=self.W, "input width should be greater or equal than W")
  assert(input:size(dimh)>=self.H, "input height should be greater or equal than H")
  
  local kW = input:size(dimw)/self.W
  local kH = input:size(dimh)/self.H
  
  self.kW = math.ceil(kW)
  self.kH = math.ceil(kH)
  self.dW = math.floor(kW)
  self.dH = math.floor(kH)
  
  --input.nn.SpatialMaxPoolingCUDA_updateOutput(self, input)
  --input.nn.SpatialMaxPooling_updateOutput(self, input)
  C['SpatialAdaptiveMaxPooling_updateOutput'](input:cdata(), self.output:cdata(), self.indices:cdata(), self.W, self.H)
  return self.output
end

function SpatialAdaptivePooling:updateGradInput(input, gradOutput)

  local dimw = 3
  local dimh = 2
  if input:nDimension() == 4 then 
    dimw = dimw + 1
    dimh = dimh + 1
  end
  
  assert(input:size(dimw)>=self.W, "input width should be greater or equal than W")
  assert(input:size(dimh)>=self.H, "input height should be greater or equal than H")
  
  local kW = input:size(dimw)/self.W
  local kH = input:size(dimh)/self.H
  
  self.kW = math.ceil(kW)
  self.kH = math.ceil(kH)
  self.dW = math.floor(kW)
  self.dH = math.floor(kH)
  
  --input.nn.SpatialMaxPoolingCUDA_updateGradInput(self, input, gradOutput)
  input.nn.SpatialMaxPooling_updateGradInput(self, input, gradOutput)
  return self.gradInput
end


function SpatialAdaptivePooling:empty()
  self.gradInput:resize()
  self.gradInput:storage():resize(0)
  self.output:resize()
  self.output:storage():resize(0)
  self.indices:resize()
  self.indices:storage():resize(0)
end
