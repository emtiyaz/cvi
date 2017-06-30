function [y, X, y_te, X_te] = get_data_gp(name, seed)
% classification data
switch name
case 'synth' 
  setSeed(seed);
  N = 5000;
  D = 2;
  s2 = .01;
  X = randn(N,D);
  D = D + 1;
  w = [0.1; -1; +1];
  eta = [ones(N,1) X]*w + s2*randn(N,1);
  y = sign(eta);
  [X_tr, y_tr, X_te, y_te] = split_data(y, X, 0.5);
  y = 2*y - 1; y_te = 2*y_te - 1;
  
case 'ionosphere'
  data = csvread('ionosphere.data-tra.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  data = csvread('ionosphere.data-tst.csv');
  X_te = data(:,1:end-1);
  y_te = data(:,end);
  clear data;
  
case 'sonar' 
  data = csvread('sonar.all-data-tra.csv');
  X = data(:,1:end-1); 
  y = data(:,end);
  data = csvread('sonar.all-data-tst.csv');
  X_te = data(:,1:end-1);
  y_te = data(:,end);
  clear data;

case 'usps_3vs5'
  load('usps_resampled');
  y = ([train_labels test_labels] + 1)/2; % 1540 obs
  X = ([train_patterns test_patterns]); 
  y = sum(bsxfun(@times, y, [0:9]'));
  idx = find(or((y==3), (y==5)));
  y = y(idx);
  y = (y==5);
  X = X(:,idx);
  
  X = X'; % 1540x256
  [N,D] = size(X);
  y = y(:); % in 0/1 encoding
  setSeed(seed);
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
  y = 2*y - 1; y_te = 2*y_te - 1;
  
case {'housing','mg', 'mpg', 'triazines', 'space_ga'}
  load(name);
  setSeed(seed);
  y = (y-mean(y))./std(y);
  X = bsxfun(@minus, X, mean(X));
  X = bsxfun(@times, X, 1./std(X));
  [X, y, X_te, y_te] = split_data(y, X, 0.5);
 
otherwise
  error('no such name');
end

function [XTr, yTr, XTe, yTe] = split_data(y, X, prop)

  N = size(y,1);
	idx = randperm(N);
  Ntr = floor(prop * N);
	idxTr = idx(1:Ntr);
	idxTe = idx(Ntr+1:end);
  XTr = X(idxTr,:);
  yTr = y(idxTr);
  XTe = X(idxTe,:);
  yTe = y(idxTe);

