function getLoss(opt, model)
   -- 2-class problem
   noutputs = 2

   ----------------------------------------------------------------------
   print '==> define loss'

   if opt.loss == 'margin' then
      -- This loss takes a vector of classes, and the index of
      -- the grountruth class as arguments. It is an SVM-like loss
      -- with a default margin of 1.

      criterion = nn.MultiMarginCriterion()
   elseif opt.loss == 'nll' then
      -- This loss requires the outputs of the trainable model to
      -- be properly normalized log-probabilities, which can be
      -- achieved using a softmax function
      model:add(nn.LogSoftMax())

      criterion = nn.ClassNLLCriterion()
   else
      error('unknown -loss')
   end
   
   return criterion
end
