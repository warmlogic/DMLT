classdef decoder < dml.method
% DECODER generative decoding.
%
%   DESCRIPTION
%   Currently only implements the GMRF model.
%
%   REFERENCE
%   
%
%   EXAMPLE
%
%   % load data
%   load 69digits
%   prm = randperm(size(X,1)); trainidx = prm(1:(end-10)); testidx = prm((end-9):end);
%   %trainidx = 1:size(X,1); testidx = 1:size(X,1);
%   msk = masks{2} | masks{3} | masks{8} | masks{9};
%   X = zscore(X(:,find(msk(masks{1}))));
%   [I,mu,sigma] = zscore(I);
%
%   % train generative decoder using raw pixel inputs
%   m = dml.gendec;
%   m = m.train(X(trainidx,:),I(trainidx,:));
%   Z = m.test(X(testidx,:));
%   for i=1:size(Z,1)
%    subplot(1,2,1);
%    imagesc(reshape(mu+Z(i,:).*sigma,[28 28])); colormap(gray);
%    subplot(1,2,2);
%    imagesc(reshape(mu+I(testidx(i),:).*sigma,[28 28])); colormap(gray); pause(1); 
%   end
%   
%   [~,idx] = sort(diag(m.Sigma),'ascend');
%   for i=1:size(m.B,2)
%     imagesc(reshape(m.B(:,idx(i)),[28 28])); colormap(gray);
%     pause;
%   end
%
%   % train generative decoder using RBM
%   load ~/Projects/69data/modeldata.mat
%   [D,mu,sigma] = zscore(modeldata);
%   m = GRBM('niter',1000,'nhidden',30,'CD',1,'p',0.15,'sparsitycost',0.5);
%   m = m.train(D); 
%
% 
%   DEVELOPER
%   Marcel van Gerven (m.vangerven@donders.ru.nl)


  properties
   
    mu; % prior mean
    
    R; % prior covariance

    encoder; % encoder needs to be specified
        
  end
  
  methods
    
    function obj = decoder(varargin)

      obj = obj@dml.method(varargin{:});

      if isempty(obj.encoder)
        error('encoder must be specified');
      end

    end
    
    function obj = train(obj,X,Y)

      % default initialization of prior; learns mean and covariance from
      % training data 
      if isempty(obj.mu)
        obj.mu = mean(Y)'; 
      end
      if isempty(obj.R)
        obj.R = cov(Y) + 1e-6*eye(size(Y,2));
      end
      
      % train from output to input; e.g. from image to brain data
      % we do not use noutput since this requires storage of everything
      
      % learn encoding model if not already done
      if isempty(obj.encoder.B)
        obj.encoder = obj.encoder.train(Y,X);
      end
      
    end
    
    function Y = test(obj,X)

      % get encoding parameters
      J = obj.encoder.filters;
      if isempty(J), J=1; end
      B = obj.encoder.B;
      b0 = obj.encoder.b0;
      Sigma = obj.encoder.Sigma;
      
      % only use non-zero columns
      idx = any(B);
      
      % only use the ones for which the variance is substantially below the
      % original variance
      %idx2 = diag(Sigma) < 0.5;
      %idx = idx & idx2';
     
      Sigma = Sigma(idx,idx);
      B = B(:,idx);
      b0 = b0(idx);
                  
      SB = Sigma \ B';
      
      iR = inv(obj.R);
      iRm = obj.R \ obj.mu;

      Y = zeros(size(X,1),size(J*B,1));
      for j=1:size(X,1)
        
        c = transpose((X(j,idx) - b0) * SB);
        
        Y(j,:) = (iR + J*B*SB*J') \ (J*c + iRm);
                
      end
      
    end

    function m = model(obj)
      % returns
      %
      

      
    end

  end
  
end
