% This code reproduces figure 1(a) in the paper.
% Written by Wu Lin and Emtiyaz Khan

clear all; close all;
% Dataset (it could be 'a1a', 'breast_cancer_scale' 'australian_scale' 'covtype_binary_scale' 'a7a')
% If you want to see quick results, please run a1a
dataset_name = 'covtype_binary_scale';
% Baselines to compare against
baselines = {'PG-exact', 'SnK-FG', 'SnK-alg2', 'Chol'};

% get data
seed = 1;
[y, X, y_te, X_te] = get_data_log_reg(dataset_name, seed);
[N, D] = size(X);

% get delta (noise precision of the prior) and algorithmic parameters
[delta, algo_params] = get_expt_params(dataset_name, 'CVI');
deltas = [1e-4; delta.*ones(D-1,1)]; % prior precision

% we will record loss vs time
log_loss = []; time = [];

%%%%%%%%
% CVI
%%%%%%%%
setSeed(1);
fprintf('CVI with stochastic gradients\n');
% initialize
tlam_1 = 0.01*ones(N,1);
tlam_2 = -0.005*ones(N,1)/2;
beta = 1/(algo_params.beta+1);
nSamples = algo_params.nSamples;
tt_cvi(1) = 0;
[tm_te,tv_te] = bayes_lin_reg(X, tlam_1, tlam_2, deltas, X_te);
log_loss_cvi(1) = compute_log_loss(y_te, tm_te, tv_te);
fprintf('%d) log-loss %.3f\n', 0, log_loss_cvi(1));

% iterate
for iter = 1:algo_params.maxItersInfer
   tic;
   % Step 4 in Alg 1: Conjugate computation
   [tm, tv] = bayes_lin_reg(X, tlam_1, tlam_2, deltas, X);

   % Step 3 in Alg 1: compute SG of the non-conjugate part
   [fb, df, dv] = E_log_p_mc(tv, {@likLogistic}, [], y, tm, nSamples);
   tlam_1 = beta*tlam_1 + (1-beta).*(df-2*(dv.*tm));
   tlam_2 = beta*tlam_2 + (1-beta).*(dv);

   % compute log_loss
   tt_cvi(iter+1) = toc;
   [tm_te,tv_te] = bayes_lin_reg(X, tlam_1, tlam_2, deltas, X_te);
   log_loss_cvi(iter+1) = compute_log_loss(y_te, tm_te, tv_te);
   fprintf('%d) log-loss %.3f\n', iter, log_loss_cvi(end));
end
log_loss{1} = log_loss_cvi(:);
time{1} = cumsum(tt_cvi(:));

%%%%%%%
% Run baselines
%%%%%%%
for i = 1:length(baselines)
   setSeed(1);
   [~, algo_params] = get_expt_params(dataset_name, baselines{i});
   algo_params.dataset_name = dataset_name;
   [nlz_b, log_loss_b, time_b] = baseline_infer(baselines{i}, y, X, deltas, y_te, X_te, algo_params);
   log_loss{i+1} = log_loss_b;
   time{i+1} = time_b;
end

%%%%%%%
% Plot
%%%%%%%
markers = {'o','s','d','*','+'};
colors = [1,0,0; 0, 1,0; 0, 0, 1; 0,0.5,0.5; 0 0 0];
for i = 1:length(baselines)+1
   semilogx(time{i}, log_loss{i}, 'marker', markers{i}, 'color', colors(i,:), 'linewidth', 3, 'markersize', 15, 'markerfacecolor', [1 1 1]);
   hold on
end

function [m,v] = bayes_lin_reg(X, tlam_1, tlam_2, deltas, X_te)
% compute predictive mean and variance of X_te given data X
% for Bayesian linear regression with ty = X*w + eps
% observations ty := tlam_1/(-2*tlam_2),
% noise variance sig2 := 1/(-2*tlam_2),
% and prior precision of w as deltas

   [N,D] = size(X);
   % convert to precision
   tlam_2 = -2*tlam_2; 

   % Cholesky of (W + X'*diag(tlam_2)*X),
   % but for numerical stability, we compute
   % Cholesky of (I + sW*X'*diag(tlam_2)*X*sW)
   sW = 1 ./ sqrt(deltas);
   XLX = X'*bsxfun(@times, X, tlam_2); %X'*diag(tlam_2)*X
   L = chol(eye(D)+(sW*sW').*XLX);

   % comput mean
   %m = X_te*(sW.*(L\(L'\(sW.*(X'*tlam_1)))));
   A = L'\(bsxfun(@times,X_te',sW));
   b = L'\(sW.*(X'*tlam_1));
   m = A'*b;

   % compute variance
   v = sum(A.*A,1)';
end

function log_loss = compute_log_loss(y, tm, tv)
% compute log loss

   p_hat = exp(likLogistic([],[], tm(:), tv(:)));
   p_hat = max(eps,p_hat); p_hat = min(1-eps,p_hat);
   err = y.*log2(p_hat) + (1-y).*log2(1-p_hat);
   log_loss = -mean(err);
end

function [delta algo_params] = get_expt_params(dataset_name, algo)
% get the parameters of the experiments

   % set delta and maximum number of runs
   switch dataset_name
   case 'a1a'
      delta = 2.8072;
      maxItersInfer=100;
   case {'a7a'}
      delta = 5;
      maxItersInfer=210;
   case {'australian_scale'}
      delta =  1e-5;
      maxItersInfer=120;
   case {'breast_cancer_scale'}
      delta =  1;
      maxItersInfer=120;
   case {'covtype_binary_scale'}
      delta =  0.0020;
      maxItersInfer=960;
   otherwise
      error('Unknown dataset name')
   end

   % set algorithmic parameters
   if nargout>1
      % set display to 1
      algo_params.display = 1;
      algo_params.compute_loss = 1;

      % wether to use monte carlo
      nSamples = 10;
      switch algo
      case {'CVI-exact', 'PG-exact', 'Chol'}
          algo_params.mc = 0;
      case {'CVI','SnK-alg2', 'SnK-FG'}
          algo_params.mc = 1;
          algo_params.nSamples = nSamples;
      otherwise
          error('do not support');
      end

      % step size and maximum number of iterations
      % TODO put a different number of iterations for each dataset
       switch dataset_name
       case {'a1a', 'a7a'}
           beta = 0.4;
           switch algo
           case {'Chol'} ;
           case {'CVI-exact', 'PG-exact'}
               algo_params.beta = beta;
           case {'CVI'}
               algo_params.beta = beta;
           case {'SnK-alg2','SnK-FG'};
               algo_params.beta = beta/(1+beta);
           end
       case {'covtype_binary_scale'}
           beta = 0.3;
           switch algo
           case {'Chol'} ;
               maxItersInfer = 50;
           case {'CVI-exact', 'PG-exact'}
               algo_params.beta = beta;
               maxItersInfer = 10;
           case {'CVI'}
               algo_params.beta = beta;
               maxItersInfer = 2;
           case {'SnK-alg2'}
               algo_params.beta = beta/(1+beta);
               maxItersInfer = 10;
           case {'SnK-FG'}
               algo_params.beta = beta/(1+beta);
               maxItersInfer = 30;
           end
       case 'australian_scale'
           beta = 0.4;
           switch algo
           case {'Chol'} ;
           case {'CVI-exact', 'PG-exact'}
               algo_params.beta = beta;
           case {'CVI'};
               algo_params.beta = beta;
           case {'SnK-alg2','SnK-FG'}
               algo_params.beta = beta/(1+beta);
           end
       case {'breast_cancer_scale'}
           beta = 0.3;
           switch algo
           case {'Chol'} ;
           case {'CVI-exact', 'PG-exact'}
               algo_params.beta = beta;
           case {'CVI'}
               algo_params.beta = beta;
           case {'SnK-alg2','SnK-FG'}
               algo_params.beta = beta/(1+beta);
           end
       end
      algo_params.maxItersInfer=maxItersInfer;
   end
end


