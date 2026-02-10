function [Eigen_Vector,Eigen_Value]=Find_K_Max_Gen_Eigen(Matrix1,Matrix2,c)

[d,d] = size(Matrix1);

% Note this is equivalent to; [V,S]=eig(St,SL); also equivalent to [V,S]=eig(Sn,St); %

 [V,S] = mexeig(Matrix1,Matrix2);

% tic
% [V,S] = eig(Matrix1,Matrix2);
% toc
% S = diag(S);
% 
  [S,index] = sort(S);
% [S, index] = sort(S, 'descend'); %降序排列

Eigen_Vector = zeros(d,c);
Eigen_Value = zeros(1,c);

p = d;
for t = 1:c
    Eigen_Vector(:,t) = V(:,index(p));
    Eigen_Value(t) = S(p);
    p = p-1;
end