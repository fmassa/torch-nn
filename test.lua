require 'cunn'

dofile 'SpatialAdaptiveMaxPooling.lua'
dofile 'SpatialAdaptiveMaxPoolingFloat.lua'

input = torch.rand(32, 3, 58, 40):cuda()

module = nn.SpatialAdaptiveMaxPooling(4,4):cuda()

out = module:forward(input)

inputF = input:float()

moduleF = nn.SpatialAdaptiveMaxPoolingFloat(4,4):float()

outF = moduleF:forward(inputF)

