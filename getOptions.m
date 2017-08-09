function opts = getOptions(method, model)
%Function used to generate the training options for the different tested
%methods

switch method
    case 'WeightDecay'
        opts.batchSize = 100;
        opts.DO = 0;
        opts.lambda = 0;
        opts.networkType = 'simplenn';
        opts.n = 0;
        opts.regType = 'none';
        switch model
            case 'model1'
                opts.WD = 0.00001;
            case 'model2'
                opts.WD = 0.0001;
            case 'model3'
                opts.WD = 0.001;
            case 'model4'
                opts.WD = 0.01;
            case 'model5'
                opts.WD = 0.1;
            case 'model6'
                opts.WD =1;
            case 'model7'
                opts.WD = 10;
            case 'model8'
                opts.WD = 100;
        end;
    case 'WD+Dropout'
        opts.batchSize = 100;
        opts.lambda = 0;
        opts.networkType = 'simplenn';
        opts.n = 0;
        opts.regType = 'none';
        switch model
            case 'model1'
                opts.WD = 0.0001;
                opts.DO = 0.1;
            case 'model2'
                opts.WD = 0.0001;
                opts.DO = 0.25;
            case 'model3'
                opts.WD = 0.0001;
                opts.DO = 0.5;
            case 'model4'
                opts.WD = 0.0001;
                opts.DO = 0.75;
            case 'model5'
                opts.WD = 0.01;
                opts.DO = 0.1;
            case 'model6'
                opts.WD =0.01;
                opts.DO = 0.25;
            case 'model7'
                opts.WD = 0.01;
                opts.DO = 0.5;
            case 'model8'
                opts.WD = 0.01;
                opts.DO = 0.75;
        end;
    case 'DataDistribution'
        opts.batchSize = 250;
        opts.WD = 0;
        opts.DO = 0;
        opts.networkType = 'simplenn_reg';
        opts.n = 0;
        opts.regType = 'data distribution';
        switch model
            case 'model1'
                opts.lambda = 0.00001;
            case 'model2'
                opts.lambda = 0.0001;
            case 'model3'
                opts.lambda = 0.001;
            case 'model4'
                opts.lambda = 0.01;
            case 'model5'
                opts.lambda = 0.1;
            case 'model6'
                opts.lambda = 1;
            case 'model7'
                opts.lambda = 10;
            case 'model8'
                opts.lambda = 100;
        end;
    case 'SliceSampling'
        opts.batchSize = 250;
        opts.WD = 0;
        opts.DO = 0;
        opts.networkType = 'simplenn_reg';
        opts.n = 0;
        opts.regType = 'slice sampling';
        
        switch model
            case 'model1'
                opts.lambda = 0.00001;
            case 'model2'
                opts.lambda = 0.0001;
            case 'model3'
                opts.lambda = 0.001;
            case 'model4'
                opts.lambda = 0.01;
            case 'model5'
                opts.lambda = 0.1;
            case 'model6'
                opts.lambda = 1;
            case 'model7'
                opts.lambda = 10;
            case 'model8'
                opts.lambda = 100;
        end;
    case 'SliceSampling2'
        opts.batchSize = 400;
        opts.WD = 0;
        opts.DO = 0;
        opts.networkType = 'simplenn_reg';
        opts.n = 6413;
        opts.regType = 'slice sampling';
        
        switch model
            case 'model1'
                opts.lambda = 0.00001;
            case 'model2'
                opts.lambda = 0.0001;
            case 'model3'
                opts.lambda = 0.001;
            case 'model4'
                opts.lambda = 0.01;
            case 'model5'
                opts.lambda = 0.1;
            case 'model6'
                opts.lambda = 1;
            case 'model7'
                opts.lambda = 10;
            case 'model8'
                opts.lambda = 100;
        end;
    case 'WD+DataDistribution'
        opts.batchSize = 250;
        opts.WD = 10^(-4);
        opts.DO = 0;
        opts.networkType = 'simplenn_reg';
        opts.n = 0;
        opts.regType = 'data distribution';
        switch model
            case 'model1'
                opts.lambda = 0.00001;
            case 'model2'
                opts.lambda = 0.0001;
            case 'model3'
                opts.lambda = 0.001;
            case 'model4'
                opts.lambda = 0.01;
            case 'model5'
                opts.lambda = 0.1;
            case 'model6'
                opts.lambda = 1;
            case 'model7'
                opts.lambda = 10;
            case 'model8'
                opts.lambda = 100;
        end;
    case 'WD+SliceSampling'
        opts.batchSize = 250;
        opts.WD = 10^(-4);
        opts.DO = 0;
        opts.networkType = 'simplenn_reg';
        opts.n = 0;
        opts.regType = 'slice sampling';
        
        switch model
            case 'model1'
                opts.lambda = 0.00001;
            case 'model2'
                opts.lambda = 0.0001;
            case 'model3'
                opts.lambda = 0.001;
            case 'model4'
                opts.lambda = 0.01;
            case 'model5'
                opts.lambda = 0.1;
            case 'model6'
                opts.lambda = 1;
            case 'model7'
                opts.lambda = 10;
            case 'model8'
                opts.lambda = 100;
        end;
end;