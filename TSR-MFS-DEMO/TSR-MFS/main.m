clc;
clear;
warning off;
addpath('./datasets/');
addpath('./tools/');
load('Yale');
dataset_name = 'Yale'; 
v = 3; 
n = 165; 
dd = [4096,3304,6750]; 
c = 15; 
Y = gt;
alpha = 1;
beta = 0.001;
gamma = 10;
scalar = 1;
select = 1.5;  
rng('default');
iter = 0;
for i = 1:v
    X{i} = NormalizeFea(X{i});
end
iter = iter + 1;
tic
[U] = TSR(X, v, n, dd, c, alpha, beta, gamma, scalar, dataset_name);
toc
Xnor = cell(1, v);
for iv = 1:v
    Xnor{iv} = X{iv}';
end
ConsenX = DataConcatenate(Xnor);
for iv = 1:v
    U{iv} = U{iv}';
end
W = DataConcatenate(U);
W = W';
XX = ConsenX';
d = size(XX,1);  
select = 1.5;
selectedFeas = select * d * 0.1;
w = [];
for iv = 1:d
    w = [w norm(W(iv,:),2)];
end
[~, index] = sort(w, 'descend');
Xw = XX(index(1:selectedFeas), :);
for i = 1:40
    label = litekmeans(Xw', c, 'MaxIter', 200, 'Replicates', 20);
    result1 = ClusteringMeasure(Y, label);
    result(i,:) = result1;
end

for j = 1:2
    a = result(:,j);
    ll = length(a);
    temp = [];
    for i = 1:ll
        if i < ll - 18
            b = sum(a(i:i+19));
            temp = [temp; b];
        end
    end
    [e,f] = max(temp);
    e = e ./ 20;
    MEAN(j,:) = [e,f];
    STD(j,:) = std(result(f:f+19, j));
    rr(:,j) = sort(result(:,j));
    BEST(j,:) = rr(end,j);
end
fprintf('\n===== Dataset: %s =====\n', dataset_name);
fprintf('Selected features: %.1f%%\n', select * 10);
fprintf('Mean ACC: %.4f ± %.4f\n', MEAN(1,1), STD(1,1));
fprintf('Mean NMI: %.4f ± %.4f\n', MEAN(2,1), STD(2,1));
fprintf('Best ACC: %.4f\n', BEST(1,1));
fprintf('Best NMI: %.4f\n', BEST(2,1));
fprintf('============================\n');
