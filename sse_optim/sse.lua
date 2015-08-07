ffi = require("ffi")
-- Load myLib
--myLib = ffi.load(paths.cwd() .. '/libsse_optim_static.a')
myLib = ffi.load(paths.cwd() .. '/libsse_optim.so')
-- Function prototypes definition
ffi.cdef [[
void sseMul(THDoubleTensor* t1, double v);
void Mul(THDoubleTensor* t1, double v);
void parMul(THDoubleTensor* t1, double v);
void parsseMul(THDoubleTensor* t1, double v);
]]

function sseMul(t,v)
  myLib.sseMul(t:cdata(),v)
end

function Mul(t,v)
  myLib.Mul(t:cdata(),v)
end
function parMul(t,v)
  myLib.parMul(t:cdata(),v)
end
function parsseMul(t,v)
  myLib.parsseMul(t:cdata(),v)
end
