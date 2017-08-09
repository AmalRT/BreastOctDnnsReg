function [net, info] = cnn_OCT(n, DoRate, poolType,dataPath, varargin)
%CNN_OCT  Demonstrates MatConvNet on OCT
% Taken from MatConvNet, modified to incorporate:
%    use of function norm regularization
%    model selection


opts.modelType = 'lenet' ;
opts.networkType = 'simplenn_reg' ;
opts.lambda = 0.005;
opts.regType = 'data distribution';
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
opts.weightDecay = 0.0001;
opts.batchSize = 145;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
opts.expDir = fullfile(vl_rootnn, 'data', ['sl-baseline-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin);
if ~isfield(opts.train, 'gpus'), opts.train.gpus = [1] ; end;

% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

net = cnn_OCT_init(DoRate, poolType,'no-cifar','networkType', opts.networkType) ;
if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath);
else
    imdb = load(dataPath);    
end
opts = vl_argparse(opts, varargin);


if strcmp(opts.networkType , 'simplenn_reg') && strcmp(opts.regType , 'slice sampling')
    imdb.images.set(end+1:end+n) =4*ones(1,n);
end;
imdb.images.data = single(imdb.images.data);
imdb.images.data_mean = single(imdb.images.data_mean);
net.meta.classes.name = imdb.meta.classes(:)' ;


% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

disp(['Norm. Otions:', num2str(double(opts.batchNormalization))]) ;  

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train;
  case 'simplenn_reg', trainfn = @cnn_train_reg;
  case 'dagnn', trainfn = @cnn_train_dag ;
end

if strcmp(opts.networkType, 'simplenn_reg')
    [net, info] = trainfn(net, imdb, getBatch(opts), ...
        'expDir', opts.expDir, ...
        net.meta.trainOpts, ...
        opts.train, ...
        'val', find(imdb.images.set == 3),'lambda', opts.lambda, 'regType', opts.regType, 'weightDecay', opts.weightDecay, 'batchSize', opts.batchSize ) ;
else
    [net, info] = trainfn(net, imdb, getBatch(opts), ...
        'expDir', opts.expDir, ...
        net.meta.trainOpts, ...
        opts.train, ...
        'val', find(imdb.images.set == 3), 'weightDecay', opts.weightDecay,'batchSize', opts.batchSize ) ;
end;

% --------------------------------------------------------------------
function fn = getBatch(opts)
% --------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'simplenn_reg'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end

% --------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;

% --------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% --------------------------------------------------------------------
images = imdb.images.data(:,:,:,batch) ;
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'input', images, 'label', labels} ;
