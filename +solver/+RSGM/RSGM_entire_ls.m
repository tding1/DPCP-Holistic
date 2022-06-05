function [B, elapsed_time, iter] = RSGM_entire_ls(Xtilde, c, maxiter, tol, mu_0, beta)

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

if c == 1
    [B, elapsed_time, iter] = solver.RSGM.RSGM_one(Xtilde);
else
    mu_min = 1e-15;

    [D, N] = size(Xtilde);
    d = D-c;

    if D <= N
        [U,~,~] = svd(Xtilde, 'econ');
    else
        [U,~,~] = svd(Xtilde);
    end
    B = U(:, (d+1 : end));
   
    mu = mu_0;
    BTX = B'*Xtilde;
    denum = sqrt(sum((BTX).^2));
    obj_old = sum( denum );
    
    Delta_J = Inf;
    i = 0;
    while mu > mu_min && Delta_J > tol && i < maxiter
        i = i+1;
        
        denum(denum==0) = inf;
        X_de = Xtilde ./ denum;
        grad = X_de * BTX';
        BTX_de = BTX ./ denum;
        grad = grad - B*(BTX_de * BTX');
        
        Btmp = orth(B - mu*grad);
        BtmpTX = Btmp' * Xtilde;
        denum_tmp = sqrt( sum( (BtmpTX).^2 ) );
        obj_tmp = sum( denum_tmp );
        while obj_tmp > obj_old && mu > mu_min
            mu = mu*beta;
            Btmp = orth(B - mu*grad);
            BtmpTX = Btmp' * Xtilde;
            denum_tmp = sqrt( sum( (BtmpTX).^2 ) );
            obj_tmp = sum( denum_tmp );
        end

        obj_new = obj_tmp;
        Delta_J = abs(obj_old-obj_new)/(abs(obj_old)+10^(-9));
        obj_old = obj_new;
        
        B = Btmp;
        BTX = BtmpTX;
        denum = denum_tmp;
    end

    iter = i;
    elapsed_time = toc(t_start);
    
end


end



