classdef encoder < dml.method
% ENCODER encode data using an encoding model
%
%   DESCRIPTION
%   An encoder takes 
%   - input data
%   - an encoding model which transforms the input data to feature space
%   - a regression algorithm in order to predict output data from features
%
%   REFERENCE
%   Kriegeskorte book
%
%   EXAMPLE
%   see encode_example
%
%   DEVELOPER
%   Marcel van Gerven (m.vangerven@donders.ru.nl)


  properties
   
    encmodel = ''; % empty encoding model takes raw pixels as input
    
    regressor = ''; % empty regression algorithm uses standard linear regression

    filters; % filters given by the encoding model
    
    Sigma; % variances
    
    B; % regressor weights
    
    b0; % regressor bias
    
    m % regression model
    
  end
  
  methods
    
    function obj = encoder(varargin)

      obj = obj@dml.method(varargin{:});

    end
    
    function obj = train(obj,X,Y)
      % Here, the input is the image data and the output is the brain data

      % squeeze the input through the encoding model
      % PREVENTED FOR PLS SINCE THERE THE FILTERS ARE LEARNED
      if ~strcmp(obj.encmodel,'') && ~any(strcmp(obj.regressor,{'opls' 'sopls'}))
        [X,obj.filters] = obj.encode(X);
      end
      
      N = size(Y,1); % number of samples
      nvoxels = size(Y,2);
      nfeatures = size(X,2);
        
      % select regression algorithm
      switch obj.regressor
       
        case 'ridge'
          
          reg = dml.glmnet('family','gaussian','alpha',0,'verbose',true,'lambda',1e-3);
       
        case 'elastic'
          
          reg = dml.glmnet('family','gaussian','alpha',0.5,'verbose',true,'validator',dml.crossvalidator('type','split','stat','-RMS'));
          
        case 'coupled'
          
          res = sqrt(size(X,2));
          v = dml.enet.lambdapath(X,nanmean(Y,2),'gaussian',50,1e-3);
          K = dml.prior([res res],[-10 -10]);
          reg = dml.gridsearch('validator',dml.crossvalidator('type','split','stat','-RMS','mva',dml.enet('L2',K,'family','gaussian','restart',false)),'vars','L1','vals',v);
          
        case {'opls','sopls','csopls'}
          
          if strcmp(obj.regressor,'opls')
            
            m = dml.sopls('method','opls','restart',false);

          elseif strcmp(obj.regressor,'sopls')
          
            m = dml.sopls('method','sopls','restart',false,'verbose',true);
          
          else
            
            res = sqrt(size(X,2));
            v = dml.enet.lambdapath(X,nanmean(Y,2),'gaussian',50,1e-3);
            K = dml.prior([res res],[-10 -10]);
            e = dml.gridsearch('verbose',true,'validator',dml.crossvalidator('type','split','stat','-RMS','mva',dml.enet('L2',K,'family','gaussian','restart',false)),'vars','L1','vals',v);
            m = dml.sopls('method','sopls','restart',false,'verbose',true,'regularizer',e);
            
          end
          
          % use MEAN -RMS error as criterion
          reg = dml.gridsearch('validator',dml.crossvalidator('type','split','stat','-RMS','mva',m),'vars','nhidden','vals',20:-1:1,'verbose',true);
          reg = reg.train(X,Y);
          
          obj.B = reg.mva.method{1}.Q';
          obj.filters = reg.mva.method{1}.P;
          obj.b0 = reg.model.bias;
          obj.Sigma = zeros(nvoxels,nvoxels);
          
          df = 1; % depends on number of free parameters
          obj.Sigma(1:(nvoxels+1):end) = sum((reg.test(X) - Y).^2) / (N-df);
          
        otherwise
  
          reg = dml.glmnet('family','gaussian','alpha',0,'lambda',1e-6);
      
      end

      if ~any(strcmp(obj.regressor,{'opls' 'sopls' 'csopls'}))
      
        % train from output to input; e.g. from image to brain data
        % we do not use noutput since this requires storage of everything
        obj.Sigma = zeros(nvoxels,nvoxels);
        obj.B = zeros(nfeatures,nvoxels);
        obj.b0 = zeros(1,nvoxels);
        
        for i=1:nvoxels
          
          if obj.verbose
            fprintf('training voxel %d of %d\n',i,nvoxels);
          end
          
          enc = reg.train(X,Y(:,i));
          
          % compute degrees of freedom
          df = 1;
          
          % compute variance on training set
          obj.Sigma(i,i) = sum((enc.test(X) - Y(:,i)).^2) / (N - df);
          
          % get all encoder weights; assumes model.weights is defined and
          % assumes that the encoder makes use of noutput to generate n outputs
          obj.B(:,i) = enc.model.weights;
          
          % get all biases; assumes m.bias is defined
          obj.b0(i) = enc.model.bias;
          
        end
        
      end
      
      obj.m = reg; % expensive; useful for debugging
      
    end
    
    function Y = test(obj,X)
      
      % squeeze the input through the encoding model
      if ~isempty(obj.filters)
        X = obj.encode(X);
      end
      
      nvoxels = size(obj.B,2);
      for i=1:nvoxels
        
        if obj.verbose
          fprintf('testing voxel %d of %d\n',i,nvoxels);
        end
        
        Y = bsxfun(@plus,X * obj.B,obj.b0);
        
      end
          
    end

    function m = model(obj)
      % returns
      % m.weights regression coefficients per output
      
      m.weights = obj.B;
      
    end
    
    function [X,filters] = encode(obj,X)
      % apply encoding model
      
      filters = obj.filters;
      switch obj.encmodel
        
        case 'gaussian'
        
          [X,gaus,indices,info,filters] = applymultiscalegaussianfilters(X,[1 2 4 8],1,.01,0);
          
        case 'gabor'
          
          cpfovs = [1 2 4 8 16];
          bandwidths = -1;
          spacings = 2;
          numor = 8;
          numph = 2;
          thresh = 0.01;
          scaling = 1;
          mode = 0;
          [X,gbrs,gaus,sds,indices,info,filters] = applymultiscalegaborfilters(X,cpfovs,bandwidths,spacings,numor,numph,thresh,scaling,mode);
          
        otherwise
          
          % just squeeze through the filters
          X = X * filters;
          
      end
      
      % check filters
      %for i=1:size(filters,2), imagesc(reshape(filters(:,i),[28 28])); pause; end
    
    end
    
    function plot_filters(obj)
      % plot linear combination of the filters for each voxel
      
      if ~isempty(obj.filters)
        F = obj.filters * obj.B;
      else
        F = obj.B;
      end
      res = sqrt(size(F,1));
      for i=1:size(F,2)

        m = max(abs(F(:,i)));
        if m
          imagesc(reshape(F(:,i),[res res]),[-m m]);
        else
          imagesc(reshape(F(:,i),[res res]));
        end
        axis square;
        colormap(gray);
        drawnow;
        pause(0.5);
        
      end
      
    end
    

  end
  
end
