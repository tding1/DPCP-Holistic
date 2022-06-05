function [P, elapsed_time, iter] = RSGM_entire_inv_ls(Xtilde, c, maxiter, tol, mu_0, beta)
% Switch to find d-dimensional (d <= N << D) subspace since c = D - d >> 0
% The problem is now:
%               min_P || Xtilde' * (I - P * P') ||_{1,2}, where P is D-by-d
% instead of:
%               min_B || Xtilde' * B ||_{1,2}, where B is D-by-c
 
t_start = tic;

if nargin < 3
    maxiter = 200;
    tol = 1e-8;
    mu_0 = 1e-2;
    beta = 0.5;
end
if nargin < 4
    tol = 1e-8;
    mu_0 = 1e-2;
    beta = 0.5;
end
if nargin < 5
    mu_0 = 1e-2;
    beta = 0.5;
end
if nargin < 6
    beta = 0.5;
end

[D, N] = size(Xtilde);
assert(D > N)  

d = D-c;

mu_min = 1e-15;

[U,~,~] = svd(Xtilde, 'econ');
P = U(:, 1:d);

mu = mu_0;
PTX = P' * Xtilde;
denum = sqrt( sum( (Xtilde - P * PTX).^2 ) );
obj_old = sum( denum );

Delta_J = Inf;
i = 0;
while mu > mu_min && Delta_J > tol && i < maxiter
    i = i+1;

    denum(denum==0) = inf;
    X_de = Xtilde ./ denum;
    grad = - X_de * PTX';
    PTX_de = PTX ./ denum;
    grad = grad + P*(PTX_de * PTX');

    Ptmp = orth(P - mu*grad);
    PtmpTX = Ptmp' * Xtilde;
    denum_tmp = sqrt( sum( (Xtilde - Ptmp * PtmpTX).^2 ) );
    obj_tmp = sum( denum_tmp );
    while obj_tmp > obj_old && mu > mu_min
        mu = mu*beta;
        Ptmp = orth(P - mu*grad);
        PtmpTX = Ptmp' * Xtilde;
        denum_tmp = sqrt( sum( (Xtilde - Ptmp * PtmpTX).^2 ) );
        obj_tmp = sum( denum_tmp );
    end

    obj_new = obj_tmp;
    Delta_J = abs(obj_old-obj_new)/(abs(obj_old)+10^(-9));
    obj_old = obj_new;

    P = Ptmp;
    PTX = PtmpTX;
    denum = denum_tmp;
end

iter = i;
elapsed_time = toc(t_start);


end

