function[U] = TSR(X, v, n, dd, c,alpha, beta,gamma,scalar,dataset_name)
MaxIter =15 ;
U = cell(1,v);
Z = cell(1,v);
Zt = cell(1,v);
Q = cell(1,v);  
for  i = 1:v
    Q{i} = eye(dd(i));
    U{i} = ones(dd(i),c);
    Z{i} = ones(n,n);
end
options = [];
options.NeighborMode = 'KNN';
options.k = 5;
options.WeightMode = 'HeatKernel';
options.t = 10;
for i = 1:v
    A{i} = constructW(X{i}',options);
end
S = zeros(n,n);
for i = 1:v
    S = S + A{i};
end
S = S/v;
sum_S = (S+S')*0.5;
Ls = diag(sum(sum_S))-sum_S;
Ls = (Ls+Ls')*0.5;
J = cell(1,v);
O = cell(1,v);
for i = 1:v
    J{i} = zeros(n,n);
    O{i} = zeros(n,n); 
end
sX = [n, n, v];
w = 1/v*ones(1,v);
pho = 2;
mu = 2;
mu_min  = 1e-8;
mu_max  = 1e8;
for iter = 1:MaxIter
    for iterv = 1:v
        LG = (eye(n) - Z{iterv});
        LG = LG * LG';
        LG = (LG + LG') / 2;
        [Y, ~, ~]=eig1(LG, c, 0);
        U{iterv}=(X{iterv}*X{iterv}'+alpha*Q{iterv})\(X{iterv}*Y);
        Ui=sqrt(sum(U{iterv}.*U{iterv},2)+eps);
        diagonal=0.5./Ui;
        Q{iterv}=diag(diagonal);
    end
    for iterv = 1:v
        tempz1 = (X{iterv}'*U{iterv}*U{iterv}'*X{iterv}+beta*Ls+w(iterv)*eye(n)+mu/2*w(iterv)*w(iterv)*eye(n)) ;
        tempz2 = (X{iterv}'*U{iterv}*U{iterv}'*X{iterv} +w(iterv)*S +mu/2*w(iterv)*J{iterv}-w(iterv)/2*O{iterv});
        tempz = (tempz1\tempz2)/ scalar;
        if(strcmp(dataset_name, 'COIL20'))
            tempz = double(tempz);
        end
        tempz = SimplexProj(tempz');
        Z{iterv} = scalar*tempz';
    end
    S = zeros(n);
   temps = zeros(n);
    Dist_all = cell(1, v);
    for iterv = 1:v
        Dist_all{iterv} = L2_distance_1(Z{iterv}', Z{iterv}');
    end
    for i=1:n
        a0 = zeros(1,n);
        for iterv = 1:v
            temp = Z{iterv};
            a0 = a0+w(1,iterv)*temp(i,:);
        end
        b0 = zeros(1,n);
        for iterv = 1:v
            b0 = b0+Dist_all{iterv}(i, :);
        end
        ad = (a0-0.25*beta*b0)/sum(w);
        temps(i,:) = EProjSimplex_new(ad);
        if(strcmp(dataset_name, 'COIL20'))
            temps = double(temps);
        end
        temps = SimplexProj(temps');
        S = scalar*temps';
    end
    sum_S = (S+S')*0.5;
    Ls = diag(sum(sum_S))-sum_S;
    Ls = (Ls+Ls')*0.5;
    for iterv=1:v
        Zt{iterv} = w(iterv)*Z{iterv};
    end
    Z_tensor = cat(3, Zt{:,:});
    O_tensor = cat(3, O{:,:});
    z = Z_tensor(:);
    o = O_tensor(:);
    [j, ~] = wshrinkObjno(z+1/mu*o,gamma/mu,sX,0,3);
    J_tensor = reshape(j, sX);
    for iterv=1:v
        J{iterv} = J_tensor(:,:,iterv);
    end
    for iterv=1:v
        O{iterv} = O{iterv} + mu*(w(iterv)*Z{iterv}-J{iterv});
    end
    for iterv = 1:v
        w(iterv) = 0.5 / (norm(Z{iterv} - S, 'fro') + eps);
    end
    mu = pho*mu;
    leq = Z_tensor-J_tensor;
    leqm = norm(abs(leq(:)));
    tensor_err = leqm;
end


