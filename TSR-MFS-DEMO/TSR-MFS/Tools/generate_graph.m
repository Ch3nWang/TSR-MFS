function [ S ] = generate_graph()

addpath('./datasets/');
load('Coil20.mat');

X=fea';

X=full(X);
d=sqrt(sum(X.^2));
X=bsxfun(@rdivide,X,d);

[n,d]=size(X);
S=cell(1, 5);
D=EuDist2(X);
optt=mean(mean(D));
clear D;

opt1.NeighborMode='KNN';
opt1.WeightMode='Binary';
opt1.k=10;
S{1}=constructW(X,opt1);

opt2.NeighborMode='KNN';
opt2.WeightMode='HeatKernel';
opt2.k=10;
opt2.t=0.1*optt;
S{2}=constructW(X,opt2);

opt3.NeighborMode='KNN';
opt3.WeightMode='HeatKernel';
opt3.k=10;
opt3.t=optt;
S{3}=constructW(X,opt3);

opt4.NeighborMode='KNN';
opt4.WeightMode='HeatKernel';
opt4.k=10;
opt4.t=10*optt;
S{4}=constructW(X,opt4);

opt5.NeighborMode='KNN';
opt5.WeightMode='Cosine';
opt5.k=10;
S{5}=constructW(X,opt5);

end

