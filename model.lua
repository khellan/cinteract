-- This is the AlexNet model from dp
function buildCudaModel(opt)
   print('Building cuda model')
   local features = nn.Concat(2)
   local fb1 = nn.Sequential() -- branch 1
   fb1:add(nn.SpatialConvolutionMM(3,48,11,11,4,4,2,2))       -- 224 -> 55
   fb1:add(nn.ReLU())
   if opt.LCN then
      fb1:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 2))
   end
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   
   fb1:add(nn.SpatialConvolutionMM(48,128,5,5,1,1,2,2))       --  27 -> 27
   fb1:add(nn.ReLU())
   if opt.LCN then
      fb1:add(inn.SpatialCrossResponseNormalization(5, 0.0001, 0.75, 2))
   end
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   
   fb1:add(nn.SpatialConvolutionMM(128,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialConvolutionMM(192,192,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialConvolutionMM(192,128,3,3,1,1,1,1))      --  13 ->  13
   fb1:add(nn.ReLU())
   
   fb1:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
   fb1:add(nn.Copy(nil, nil, true)) -- prevents a newContiguous in SpatialMaxPooling:backward()

   local fb2 = fb1:clone() -- branch 2
   for k,v in ipairs(fb2:findModules('nn.SpatialConvolutionMM')) do
      v:reset() -- reset branch 2's weights
   end

   features:add(fb1)
   features:add(fb2)

   -- 1.3. Create Classifier (fully connected layers)
   local classifier = nn.Sequential()
   classifier:add(nn.View(256*6*6))
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(256*6*6, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   
   classifier:add(nn.Dropout(0.5))
   classifier:add(nn.Linear(4096, 4096))
   classifier:add(nn.Threshold(0, 1e-6))
   
   classifier:add(nn.Linear(4096, 2)) -- Second parameter is the output layer dimension
   classifier:add(nn.LogSoftMax())

   -- 1.4. Combine 1.1 and 1.3 to produce final model
   local model = nn.Sequential():add(features):add(classifier)
   -- wrap the nn.Module in a dp.Module

   return model
end

function buildCpuModel()
   print('Building cpu model')
   -- 2-class problem
   local noutputs = 2

   -- input dimensions
   local nfeats = 3
   local width = 64
   local height = 64
   local ninputs = nfeats*width*height

   -- number of hidden units (for MLP only):
   local nhiddens = ninputs / 2

   -- hidden units, filter sizes (for ConvNet only):
   local nstates = {64,64,128}
   local filtsize = 5
   local poolsize = 2
   local normkernel = image.gaussian1D(7)
   local model = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
   model:add(nn.Tanh())
   model:add(nn.SpatialLPPooling(nstates[1],2,poolsize,poolsize,poolsize,poolsize))
   model:add(nn.SpatialSubtractiveNormalization(nstates[1], normkernel))

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
   model:add(nn.Tanh())
   model:add(nn.SpatialLPPooling(nstates[2],2,poolsize,poolsize,poolsize,poolsize))
   model:add(nn.SpatialSubtractiveNormalization(nstates[2], normkernel))

   -- stage 3 : standard 2-layer neural network
   model:add(nn.Reshape(nstates[2]*filtsize*filtsize))
   model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
   model:add(nn.Tanh())
   model:add(nn.Linear(nstates[3], noutputs))

   return model
end

function buildModel()
   if opt.cuda then
      return buildCudaModel()
   end
   return buildCpuModel()
end
