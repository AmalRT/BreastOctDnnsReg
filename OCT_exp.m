%Script used to train and test the different models.
%The script select the best model based on the best AUC and test is on the
%test data. 
%% Setup :
 
addpath(genpath('matconvnet-master'));
vl_compilenn('enableGpu', true);


%% Global parameters:
dataFolder = '/home/amal-rt/Documents/Back-up-2/OCT_NN/examples/OCT/';
testFolder = '/home/amal-rt/Documents/Back-up-2/OCT_NN/DataNN/Test/';
expFolder = '/home/amal-rt/Documents/Back-up-2/OCT_NN/examples/Experiments/';
dataPath = {'imdb.mat','imdb1.mat','imdb2.mat','imdb3.mat','imdb4.mat','imdb5.mat'};
dataConfig = {'a-c', 'b-c', 'c-a', 'c-b'};
methods = { 'WeightDecay', 'WD+Dropout', 'DataDistribution', 'SliceSampling'};
models = {'model1','model2','model3','model4', 'model5', 'model6','model7', 'model8'};
pooltypes = 'max';

%% Prepare environement and data:
mkdir(expFolder);

for k=1:4
    disp(['Beginning configuration: ',dataConfig{k} ]);
    dataFolder_k = [dataFolder,dataConfig{k},'/' ];
    testFolder_k = [testFolder,dataConfig{k},'/' ];
    mkdir(dataFolder_k);
    % Extract patches
    if k>1
    if k==1
       % The extraction of regularization samples need to be done once.
       test_time1=extractPatchFnc(dataConfig{k},0,0,1);
    else
        test_time1=extractPatchFnc(dataConfig{k},0,0,0);
    end;
    %Construct data base:
    imdbsConstructFnc(5,dataFolder_k,dataConfig{k},0);
    end;
    
    % Train the different methods
    for i = 1:6
        disp(['Beginning ', methods{i}]);
        auc_k = zeros(8,5);
        
     % Cross-validation for parameter selection   
        for j = 1:8
            disp(['Beginning ', methods{i},', ', models{j}])
            expDir = [expFolder,dataConfig{k},'/', methods{i}, '/', models{j}];
            mkdir(expDir);
            opts = getOptions(methods{i},models{j});
            for m = 1:5
                dataPath_m = [dataFolder_k,dataPath{m+1}];
                mkdir([expDir,'/', num2str(m)]);
                try
                    [net_OCT, info_OCT] = cnn_OCT(opts.n,opts.DO, pooltypes,dataPath_m,...
                        'expDir', [expDir,'/', num2str(m)], ...
                        'networkType', opts.networkType, 'lambda', opts.lambda, 'regType', opts.regType, 'weightDecay', opts.WD, 'batchSize', opts.batchSize);
                    [labels,scores] = valFnc(net_OCT,dataPath_m);
                    [~,~,~,auc_k(j,m)]= perfcurve(labels,scores(:,1),1);
                catch
                    disp('Training procedure failed');
                end;
                
            end;
            
            
        end;
        save([expFolder,dataConfig{k},'/', methods{i},'/auc.mat'],'auc_k');
        disp('Model selection and trainning on the whole data');
        % Model selection
        auc_mean = mean(auc_k,2);
        [~,model_sel] =max(auc_mean);
        % Train the selected model
        expDirFinal = [expFolder,dataConfig{k},'/', methods{i}, '/Final_model'];
        mkdir(expDirFinal);
        optsFinal = getOptions(methods{i}, models{model_sel});
        dataPathFinal = [dataFolder_k,dataPath{1}];
        [net_OCT_Final, info_OCT_Final] = cnn_OCT(optsFinal.n,optsFinal.DO, pooltypes,dataPathFinal,...
            'expDir', expDirFinal, ...
            'networkType', optsFinal.networkType, 'lambda', optsFinal.lambda, 'regType', optsFinal.regType, 'weightDecay', optsFinal.WD, 'batchSize', optsFinal.batchSize);
        
        % Test with the selected model
        disp('Getting test results');
        testDir = [expFolder,dataConfig{k},'/', methods{i}, '/Test_res'];
        mkdir(testDir)
        [labels, scores,test_time2] = testFnc(net_OCT_Final, testFolder_k);
        test_time= test_time1+test_time2;
        save([testDir,'/testRes.mat'], 'labels','scores','test_time','model_sel');
        disp(['End of ', methods{i}]);
    end;
    disp(['End of configuration: ',dataConfig{k} ]);
end;




