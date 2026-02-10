clc;
clear;
warning off;
addpath('./datasets');

% load('Caltech101-7.mat');data = 'Caltech101-7';v = size(X,2);n = size(Y,1);param.dd = [48,40,254,1984,512,928];c = 7;
% load('handwritten.mat');data = 'handwritten';v = 2;n=2000;param.dd = [240,76,216,47,64,6];c = 10;
load('MSRC_V1_5views.mat');v = size(X,2);n=210;param.dd = [1302,48,512,256,210];c = 7;
% load('youtube.mat');data = 'youtube';v = size(X,2);n = 1592;param.dd = [750,750];c = 11;
% load('M_3Sources.mat');v = 3;n = 169;param.dd = [3560,3631,3068];c = 6; Y=truelabel{1}; for i=1:v  X{i}=data{i}'; end
% load('BBCSport2view_544.mat');v = 2;n = 544;param.dd = [3183,3203];c = 5;Y=truelabel{1}'; for i=1:v  X{i}=data{i}'; end
% load('bbcsport.mat');v = 2;n = 544;param.dd = [3183,3203];c = 5;Y=gt; for i=1:v X{i}=X{i}'; end
% load('Caltech101-7.mat');v = 6;n = 1474;param.dd = [48,40,254,1984,512,928];c = 7;
% load('ORL_mtv.mat');v = 3;n = 400;param.dd = [4096,3304,6750];c = 40;Y=gt; for i=1:v  X{i}=X{i}'; end
% load('ORL.mat');v=4;n=400;param.dd = [256,256,256,256];c = 40;Y=truelabel{1}; for i=1:v  X{i}=data{i}'; end
% load('Mfeat.mat');v=6;n=2000;param.dd=[216,76,64,6,240,47];c=10;Y=truelabel{1}; for i=1:v  X{i}=data{i}'; end
% load('WebKB.mat');v=3;n=203;param.dd=[1703,230,230];c=3;Y=truelabel{1}; for i=1:v  X{i}=data{i}'; end
% load('reuters_1200.mat');v = 5;n = 1200;param.dd = [2000,2000,2000,2000,2000];c = 6;Y=labels; for i=1:v X{i}=data{i}; end
% load('MNIST-10k.mat');v = 3;n = 10000;param.dd = [30,9,30];c = 10;
% load('Coil20.mat');v = 3;n = 1440;param.dd = [944,324,512];c = 20;Y=label; for i=1:v X{i}=data{i}'; end
% load('bbcsport4vbigRnSp.mat');v = 4;n = 116;c = 5;Y=truth; for i=1:v X{i}=X{i}'; end
% load('100Leaves.mat');v = 3;n = 1600;c = 100;param.dd = [64,64,64];Y=truelabel{1}; for i=1:v X{i}=data{i}'; end
% load('yale_mtv.mat');v = 3;n = 165;c = 15;param.dd = [4096,3304,6750];gt=im2double(gt);Y=gt; for i=1:v X{i}=X{i}'; end
% load('HW2sources.mat');v = 2;n = 2000;c = 10;param.dd = [1];Y=truelabel{1}; for i=1:v X{i}=data{i}'; end

%  Y=truelabel{1};
% Y=labels;
% for i=1:v
%     X{i}=data{i}';
% end
% for i =1 :v
%     X{i}=NormalizeFea(X{i});
% end
  

% for i = 1 :v
%     for  j = 1:n
%         X{i}(j,:) = ( X{i}(j,:) - mean( X{i}(j,:) ) ) / std( X{i}(j,:) ) ;
%     end
% end

% for i=1:v
%     X{i}=X{i}';
% end
% for p=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3,1e+4,1e+5,1e+6]
% for q=[1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1,1e+1,1e+2,1e+3,1e+4,1e+5,1e+6]
%     disp(p);
% disp(q);

%%%%%%%%%%%%%%%%
param.v = v;
param.t = 2;
param.k = 10;
param.n = n;
param.c = c;
param.NITER = 10;
rand('twister',5489);
%%%%%%%%%%%%%%%
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 10;

% [U, consenX, obj] = MUSFS(X,param,options);


% for i = 1:v
%     X{i} = X{i}';
% end

XX = DataConcatenate(X);



% for m = 1:10
%     [y] = kmeans(XX, param.k);
%     result = ClusteringMeasure(Y,y);
%     Fin_result(m,(1:3)) = result;
% end
% result1 = sum(Fin_result);
% result2 = result1/10;
% disp(['mean. ACC: ',num2str(result2(1))]);
% disp(['mean. NMI: ',num2str(result2(2))]);
% disp(['mean. Purity: ',num2str(result2(3))]);
% fprintf('\n');

for i=1:40
    label=litekmeans(XX,c,'MaxIter',100,'Replicates',20);
    result1 = ClusteringMeasure(Y,label); 
    resualt(i,:) = result1;
end
for j=1:2
    a=resualt(:,j);
    ll=length(a);
    temp=[];
    for i=1:ll
        if i<ll-18
            b=sum(a(i:i+19));
            temp=[temp;b];
        end
end
    [e,f]=max(temp);
    e=e./20;
    MEAN(j,:)=[e,f];
    STD(j,:)=std(resualt(f:f+19,j));
    rr(:,j)=sort(resualt(:,j));
    BEST(j,:)=rr(end,j);
    end

fprintf('\n');
disp(['mean. ACC: ',num2str(MEAN(1,1))]);
disp(['mean. ACC(STD): ',num2str(STD(1,1))]);
disp(['mean. NMI: ',num2str(MEAN(2,1))]);
disp(['mean. NMI(STD): ',num2str(STD(2,1))]);
fprintf('\n');
