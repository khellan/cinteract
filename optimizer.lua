function getOptimizer(opt)
   if opt.optimization == 'CG' then
      optimState = {
         maxIter = opt.maxIter
      }
      optimMethod = optim.cg

   elseif opt.optimization == 'LBFGS' then
      optimState = {
         learningRate = opt.learningRate,
         maxIter = opt.maxIter,
         nCorrection = 10
      }
      optimMethod = optim.lbfgs

   elseif opt.optimization == 'SGD' then
      optimState = {
         learningRate = opt.learningRate,
         weightDecay = opt.weightDecay,
         momentum = opt.momentum,
         learningRateDecay = 1e-7
      }
      optimMethod = optim.sgd

   elseif opt.optimization == 'ASGD' then
      optimState = {
         eta0 = opt.learningRate,
         t0 = trsize * opt.t0
      }
      optimMethod = optim.asgd

   else
      error('unknown optimization method')
   end
   
   return optimState, optimMethod
end