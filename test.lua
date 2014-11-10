require 'cunn'

dofile 'SpatialAdaptiveMaxPooling.lua'

input = torch.Tensor(32, 3, 58, 40):cuda()

module = nn.SpatialAdaptivePooling(4,4):cuda()

print(module:forward(input):size())

