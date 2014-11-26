local LocalResponseNormalization, parent = torch.class('nn.LocalResponseNormalization', 'nn.Module')

function LocalResponseNormalization:__init(size, alpha, beta)
  parent.__init(self)
  
  self.size = size or 3
  self.alpha = alpha or 0.00005
  self.beta = beta or 0.75
  
  local pad = math.floor(self.size/2)
  self.padder =  nn.SpatialZeroPadding(pad, pad, pad, pad):cuda()
  
  self.m1 = nn.Sequential()
  self.m1:add(nn.SpatialZeroPadding(pad, pad, pad, pad))
  self.m1:add(nn.Power(2))
  
  self.nInputPlanes = 0
  self.output = torch.Tensor()
    
end

function LocalResponseNormalization:makeKernel(ndimd)
  local kernel = torch.Tensor(ndimd,ndimd,self.size,self.size):fill(0)
  for i=1,ndimd do
    kernel[i][i]:fill(self.alpha/(self.size^2))
  end
  kernel = kernel:reshape(ndimd,ndimd*self.size*self.size)
  
  self.conv = nn.SpatialConvolutionMM(ndimd,ndimd,self.size,self.size)
  self.conv.weight = kernel
  self.conv.bias:fill(1)
  
  local numerator = nn.Identity()
  local denominator = nn.Sequential()
  
  local pad = math.floor(self.size/2)
  denominator:add(nn.SpatialZeroPadding(pad, pad, pad, pad))
  denominator:add(nn.Power(2))
  denominator:add(self.conv)
  denominator:add(nn.Power(self.beta))
  
  local divide = nn.ConcatTable()
  divide:add(numerator)
  divide:add(denominator)
  
  self.modules = nn.Sequential()
  self.modules:add(divide)
  self.modules:add(nn.CDivTable())
  
  self.recompKernel = false
  
  --return kernel
end

function LocalResponseNormalization:updateOutputNew(input)
  
  local dimd = 1
  if input:nDimension() == 4 then 
    dimd = dimd + 1
  end

  if self.nInputPlanes ~= input:size(dimd) or self.recompKernel then
    self.nInputPlanes = input:size(dimd)
    self:makeKernel(input:size(dimd))
    
    self.m1 = self.m1:type(input:type())
    self.conv = self.conv:type(self.output:type())
    
    
    self.modules = self.modules:type(self.output:type())
    
    self.recompKernel = false
  end
  
  self.output:resize(input:size())
  
  --[[
  local squareinput = self.m1:forward(input)
  self.m1:get(1).output = torch.Tensor():type(self.output:type())
  self.m1:get(2).output = torch.Tensor():type(self.output:type())
  local c = self.conv:forward(squareinput):pow(self.beta)
  self.conv.output = torch.Tensor():type(self.output:type())
  self.conv.gradInput = torch.Tensor():type(self.output:type())
  self.output:cdiv(input,c)
  ]]
  
  --self.output = self.modules:forward(input)
  
  if input:nElement() > 256*450*450 then
    self.m1 = self.m1:float()
    self.conv = self.conv:float()
    local squareinput = self.m1:forward(input:float())
    self.m1:get(1).output = torch.Tensor():float()
    self.m1:get(2).output = torch.Tensor():float()
    local c = self.conv:forward(squareinput):pow(self.beta)
    self.conv.output = torch.Tensor():float()
    self.conv.gradInput = torch.Tensor():float()
    self.output:cdiv(input,c:typeAs(self.output))
    --self.modules = self.modules:float()
    --self.output = self.modules:forward(input:float()):type(self.output:type())
    self.recompKernel = true
  else
    local squareinput = self.m1:forward(input)
    self.m1:get(1).output = torch.Tensor():type(self.output:type())
    self.m1:get(2).output = torch.Tensor():type(self.output:type())
    local c = self.conv:forward(squareinput):pow(self.beta)
    self.conv.output = torch.Tensor():type(self.output:type())
    self.conv.gradInput = torch.Tensor():type(self.output:type())
    self.output:cdiv(input,c)
    --self.output = self.modules:forward(input)
  end
  
  --[[
  if input:nDimension() == 4 then
    for i=1,input:size(1) do
      --self.output[i] = self.modules:forward(input[i])
      --local s = input[i]:nElement()
      --s = s[2]>s[3] and s[3] or s[2]
      if input[i]:nElement() > 256*200*200 then
        self.modules = self.modules:float()
        self.output[i] = self.modules:forward(input[i]:float()):type(self.output:type())
        self.recompKernel = true
        --netLighter(self.modules)
        --self.modules = self.modules:type(self.output:type())
      else
        self.output[i] = self.modules:forward(input[i])
      end
      
    end
  else
    self.output = self.modules:forward(input)
  end
  ]]

  return self.output
end

function LocalResponseNormalization:updateGradInput(input,gradOutput)

  local dimd = 1
  if input:nDimension() == 4 then 
    dimd = dimd + 1
  end
  
  if self.nInputPlanes ~= input:size(dimd) then
    self.nInputPlanes = input:size(dimd)
    self:makeKernel(input:size(dimd))
    self.modules = self.modules:type(self.output:type())
  end
  
  self.gradInput:resize(input:size())
  
  self.gradInput = self.modules:backward(input,gradOutput)

  return self.gradInput

end

function LocalResponseNormalization:updateOutput(input)
  
  local dimw = 3
  local dimh = 2
  local dimd = 1
  if input:nDimension() == 4 then 
    dimw = dimw + 1
    dimh = dimh + 1
    dimd = dimd + 1
  end
  
  local squareinput = torch.pow(self.padder(input),2):float()

  local kernel = torch.Tensor(input:size(dimd),self.size,self.size):fill(1)
  
  self.output:resize(input:size())
  
  if input:nDimension() == 4 then 
    for i=1,input:size(1) do
      local c = torch.conv2(squareinput[i],kernel:float()):mul(self.alpha/(self.size^2)):add(1):pow(self.beta)
      self.output[i]:cdiv(input[i],c:typeAs(input[i]))
      --self.output[i]:cdiv(input[i]:float(),c)
    end
  else
    local c = torch.conv2(squareinput,kernel:float()):mul(self.alpha/(self.size^2)):add(1):pow(self.beta)
    self.output:cdiv(input,c:typeAs(input))
    --self.output:cdiv(input:float(),c)
  end
  
  return self.output
end
