local SpatialPyramidPooling, parent = torch.class('nn.SpatialPyramidPooling', 'nn.Module')

function SpatialPyramidPooling:__init(pyr)
  parent.__init(self)
  
  self.pyr = pyr  
  self.nLevels = #pyr
  
  self.inputDepth = 0
  self.dimd = 0
  
end

function SpatialPyramidPooling:updateOutput(input)
  
  local dimd = 1
  if input:nDimension() == 4 then 
    dimd = dimd + 1
  end

  if self.inputDepth ~= input:size(dimd) or self.dimd ~= dimd then
    self.dimd = dimd
    self.inputDepth = input:size(dimd)
    self.module = nn.Concat(dimd)
    for i=1,#self.pyr do
      local t = nn.Sequential()
      t:add(nn.SpatialAdaptivePooling(self.pyr[i][1],self.pyr[i][2]))
      t:add(nn.Reshape(self.pyr[i][1]*self.pyr[i][2]*self.inputDepth))
      self.module:add(t)
    end
  
  end
  
  self.output = self.module:updateOutput(input)
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
    self.module = nn.Concat(dimd)
    for i=1,#self.pyr do
      local t = nn.Sequential()
      t:add(nn.SpatialAdaptivePooling(self.pyr[i][1],self.pyr[i][2]))
      t:add(nn.Reshape(self.pyr[i][1]*self.pyr[i][2]*self.inputDepth))
      self.module:add(t)
    end
  
  end

  self.gradInput = self.module:updateGradInput(input,gradOutput)
  return self.gradInput
end

