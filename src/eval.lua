-------------------------------------------------------------------------------
print '==> initializing...'

local cmd = torch.CmdLine()

cmd:text()
cmd:text('tofu runnable')
cmd:text()
cmd:text('Options:')
cmd:option('-nocuda',          false,        'Whether to disable cuda')
cmd:option('-gpuID',           1,            'ID of GPUs to use')
cmd:option('-mode',            'denoise',    'Model class for evaluation')
cmd:option('-inpath',          '',           'The input sequence directory')
cmd:option('-outpath',         '',           'The location to store the result')
cmd:option('-eval_list',       '',           'List of evaluation input/output paths.')
cmd:text()  

opt = cmd:parse(arg or {})
opt.cuda = not opt.nocuda

require('main/init')
local loader = require('main/loader')
local gen_path = loader.gen_path
local get_file = loader.get_file

mode = opt.mode

if opt.inpath ~= '' then
  inpath = opt.inpath
else
  if mode == 'denoise' then
    inpath = '../data/vimeo_denoising_test/input'
  elseif mode == 'deblock' then
    inpath = '../data/vimeo_deblockin_test/input'
  elseif mode == 'sr' then
    inpath = '../data/vimeo_super_resolution_test/input'
  elseif mode == 'interp' then
    inpath = '../data/vimeo_interp_test/input'
  end
end

if opt.outpath ~= '' then
  outpath = opt.outpath
else
  if mode == 'denoise' then
    outpath = '../output/dnoise'
  elseif mode == 'deblock' then
    outpath = '../output/deblock'
  elseif mode == 'sr' then
    outpath = '../output/sr'
  elseif mode == 'interp' then
    outpath = '../output/interp'
  end
end

print('  input sequence: '..inpath)
print('  result stored in: '..outpath)
paths.mkdir(outpath)

print '==> loading...'

if mode == 'denoise' then
  modelpath = '../models/denoise.t7'
elseif mode == 'sr' then 
  modelpath = '../models/sr.t7'
elseif mode == 'interp' then
  modelpath = '../models/interp.t7'
elseif mode == 'deblock' then
  modelpath = '../models/deblock.t7'
  mode = 'denoise'
end

local loadMode = 'sep'
if mode == 'interp' then loadMode = 'tri' end
model = torch.load(modelpath):float()
if opt.cuda then 
  model = cudnn.convert(model,cudnn)
  model = model:cuda()
end
model:evaluate()

local time = 0
local timer = torch.Timer()
local sample, h, w
local counter = 0
local iter = 1

input_list = {}
output_list = {}
myfile = io.open(opt.eval_list)
if myfile then
  for line in myfile:lines() do
    local in_a, in_c, out_b = line:match("(.+), (.+), (.+)")
    input_list[iter] = {in_a, in_c}
    output_list[iter] = out_b
    iter = iter + 1
  end
else
  print 'Failed to open eval list file.'
end

-- https://stackoverflow.com/questions/9102126/lua-return-directory-path-from-path
function getDirPath(str)
    return str:match("(.*[/\\])")
end

for i = 1, iter - 1 do
  sample, h, w = get_file(input_list[i], loadMode, h, w)
  h = 16 * math.floor(h / 16)
  w = 16 * math.floor(w / 16)

  print '==> processing...'

  sample, h, w = get_file(input_list[i], loadMode, h, w)
  if opt.cuda then
    sample = ut.nn.cudanize(sample)
  else
    sample = ut.nn.floatize(sample)
  end
  timer:reset()
  local output = ut.nn.memFriendlyForward(model, sample):float()
  output = output:float()
  if mode == 'sr' then
    output:add(sample[1]:float())
  end
  output = ut.tf.defaultDetransform(output:squeeze())
  local curTime = timer:time().real
  time = time + timer:time().real

  -- Clean memory
  ut.nn.sanitize(model)
  collectgarbage()
  collectgarbage()

  -- Save results
  paths.mkdir(getDirPath(output_list[i]))
  local p2 = output_list[i]
  image.save(p2, output)

  print('  frame '..i..' is done, it takes '..curTime..'s')
  counter = counter + 1
end


print '==> finishing...'

sample = nil
print('  '..mode..' takes '..time..'s on '..counter..' sequences')
print('  '..'i.e. '..(time/counter)..'s on average per sequence')
