----------------------------------------------------------------------
-- This script trains a convnet with various loss functions alternatives.
--
-- It's based on the Torch tutorial trainer by Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('CNN Trainer')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-trainPath', '', 'Path to train set')
   cmd:option('-testPath', '', 'Path to test set')
   cmd:option('-save', 'results', 'subdirectory to save/log experiments in')
   cmd:option('-visualize', false, 'visualize input data and weights during training')
   cmd:option('-plot', false, 'live plot')
   cmd:option('-cuda', false, 'use CUDA')
   cmd:option('-height', 0, 'height of images')
   cmd:option('-width', 0, 'width of images')
   cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
   cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
   cmd:option('-batchSize', 1, 'mini-batch size (1 = pure stochastic)')
   cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
   cmd:option('-momentum', 0, 'momentum (SGD only)')
   cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
   cmd:option('-maxIter', 2, 'maximum nb of iterations for CG and LBFGS')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
   cmd:option('-epochs', 100, 'Number of epochs to run')
   cmd:text()
   opt = cmd:parse(arg or {})
end

torch.setdefaulttensortype('torch.FloatTensor')

dofile('dataset.lua')
dofile('model.lua')
dofile('optimizer.lua')
dofile('loss.lua')
dofile('train_loop.lua')
dofile('test_loop.lua')

model = buildModel(opt)

criterion = getLoss(opt, model)

if opt.cuda then
   require 'cunn'
   model:cuda()
   criterion:cuda()
end

----------------------------------------------------------------------
print '==> defining some tools'

-- Log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
if model then
   parameters, gradParameters = model:getParameters()
end

trainData, classes = loadData(opt.trainPath, opt.width, opt.height)
normalizeData(trainData)
testData, _ = loadData(opt.testPath, opt.width, opt.height)
normalizeData(testData)

optimState, optimMethod = getOptimizer(opt)
print(model)

-- This matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)


print '==> training'

currentEpoch = 0
bestEpoch = 0
bestTotalValid = 0.0
while currentEpoch < opt.epochs do
   currentEpoch = train()
   totalValid = test()
   if totalValid > bestTotalValid then
      bestEpoch = currentEpoch
      bestTotalValid = totalValid
   end
end

print('Summary:')
print('Best result: ' .. (bestTotalValid * 100) .. '%')
print('Best epoch : ' .. bestEpoch)

