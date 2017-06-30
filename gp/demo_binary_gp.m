% This code compares CVI with Prox-Grad method of Khan et. al. (UAI 2016) and EP. 
% For the first two methods, we test 3 methods to approximate E[log p(y|f)]
% 'gauss_hermite', 'piecewise', 'monte_carlo'.
% For gauss_hermite we use likKL.m from GPML toolbox.
% We use binary classification with logit likelihood, but
% the monte-carlo and gauss-hermite options work for general likelihood functions. 
%
% Written by Emtiyaz Khan (RIKEN) and Wu Lin (RIKEN).

clear all
% Choose a dataset and an approximation for E(log p(y|f))
seed = 147;
data_name = 'usps_3vs5';%, 'usps_3vs5', 'sonar', 'housing'
hyp.approx_method = 'piecewise'; % 'gauss_hermite', 'piecewise', 'monte_carlo'

% get data
[y, X, y_te, X_te] = get_data_gp(data_name, seed);

% set max_iters and hyperparameters for each data
switch data_name
case 'ionosphere'
  hyp.max_iters = 30; 
  ell = 1; 
  sf = 2.5; 
case 'usps_3vs5' 
  hyp.max_iters = 100; 
  ell = 2.5;
  sf = 5;
case 'sonar'
  hyp.max_iters = 100;
  ell = -1;
  sf = 6;
otherwise
  error('no such data name');
end

% set step size and a few other hyperparameters
hyp.verbose = 0; % set to 0 if you don't want display
switch hyp.approx_method
  case 'gauss_hermite';
    hyp.step_size = .5;
  case 'piecewise'; 
    hyp.step_size = .5; 
  case 'monte_carlo'; 
    hyp.step_size = .5; 
    hyp.nSamples = 100;
    hyp.test_convergence = 0;
    hyp.compute_marglik = 0;
  otherwise
    error('no such method');
end

% set the GP prior with covSEiso Kernel
cov_func = {@covSEiso}; 
hyp.cov = [ell; sf];
mean_func = {@meanZero}; 
hyp.mean = [];
lik_func = {@likLogistic}; 
hyp.lik = [];


% run algos 
algos = {'infKL_cvi', 'infKL_PG','infEP'}; % compare against EP
setSeed(1);
for i = 1:length(algos)
  tic;
  [~,~,m_hat,v_hat,log_p_hat,~,nlZ(i)] = gp(hyp, algos{i}, mean_func, cov_func, lik_func, X, y, X_te, y_te);
  tt(i) = toc;
  
  % compute log_loss
  log_loss(i) = -mean(log_p_hat);
  fprintf('%s, log_loss = %0.4f, nlZ = %0.4f, took %1.1fs\n', algos{i}, log_loss(i), nlZ(i), tt(i));
end


