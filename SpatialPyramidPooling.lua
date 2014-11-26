local SpatialPyramidPooling, parent = torch.class('nn.SpatialPyramidPooling', 'nn.Module')

function SpatialPyramidPooling:__init(pyr)
  parent.__init(self)
  
  self.pyr = pyr
  
  self.inputDepth = 0
  self.dimd = 0
  
  self.output = torch.Tensor()
  
end

function SpatialPyramidPooling:updateOutput(input)
  
  local dimd = 1
  local batchMode = false
  if input:nDimension() == 4 then 
    dimd = dimd + 1
    batchMode = true
  end

  if self.inputDepth ~= input:size(dimd) or self.dimd ~= dimd then
    self.dimd = dimd
    self.inputDepth = input:size(dimd)
    self.modules = nn.Concat(dimd)
    for i=1,#self.pyr do
      local t = nn.Sequential()
      if self.output:type() == 'torch.FloatTensor' then
        t:add(nn.SpatialAdaptiveMaxPoolingFloat(self.pyr[i][1],self.pyr[i][2]))
      elseif self.output:type() == 'torch.CudaTensor' then
        t:add(nn.SpatialAdaptiveMaxPooling(self.pyr[i][1],self.pyr[i][2]))
      end
      t:add(nn.Reshape(self.pyr[i][1]*self.pyr[i][2]*self.inputDepth,batchMode))
      self.modules:add(t)
    end
    self.modules = self.modules:type(self.output:type())
  end
  
  assert(input:type()==self.output:type(),'Wrong input type!')
  
  self.output = self.modules:updateOutput(input)
  return self.output
end

function SpatialPyramidPooling:updateGradInput(input, gradOutput)
  
  local dimd = 1
  if input:nDimension() == 4 then 
    dimd = dimd + 1
  end

  if self.inputDepth ~= input:size(dimd) or self.dimd ~= dimd then
    self.dimd = dimd
    self.inputDepth = input:size(dimd)
    self.modules = nn.Concat(dimd)
    for i=1,#self.pyr do
      local t = nn.Sequential()
      t:add(nn.SpatialAdaptiveMaxPoolingFloat(self.pyr[i][1],self.pyr[i][2]))
      t:add(nn.Reshape(self.pyr[i][1]*self.pyr[i][2]*self.inputDepth))
      self.modules:add(t)
    end
    self.modules = self.modules:type(self.output:type())
  end

  self.gradInput = self.modules:updateGradInput(input,gradOutput)
  return self.gradInput
end

