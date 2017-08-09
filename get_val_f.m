function y = get_val_f(x, net, sz, numGpus)
%Function used to evaluate the funtion related to a DNN. This function is
%needed for generating samples for regularization with Slice Sampling.
%Amal RANNEN TRIKI - May 2016

x = single(reshape(x,sz));
if numGpus>=1
    x = gpuArray(x);
end;
res = vl_simplenn(net,x);
y = res(end).x;
if size(y,1) == 1 && size(y,2)==1
    y = reshape(y, size(y,3), size(y,4));
end;
if numGpus>=1
    y = gather(y);
end;