function imReg = get_sample(net,N,sz, numGpus, initial)
% Function used to get samples for DNN function norm regularizatin using
% slice sampling

% Amal RANNEN TRIKI - May 2016

f = @(x) get_val_f(x,net,sz,numGpus);
ff = @(x) f(x)'*f(x);
imReg = slicesample(initial, N, 'pdf', ff, 'burnin', 100);%, 'thin', 2);
sz = [sz, size(imReg,1)];
imReg = single(reshape(imReg, sz));
