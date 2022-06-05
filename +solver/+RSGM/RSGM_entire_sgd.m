function [B, elapsed_time, iter] = RSGM_entire_sgd(Xtilde, c, maxiter, mu_0, beta, batch_size)

t_start = tic;

[D, N] = size(Xtilde);
d = D-c;

if nargin < 3
    maxiter = 200;
    mu_0 = 1e-2;
    beta = 0.5;
    batch_size = min(D, ceil(N/16));
end
if nargin < 4
    mu_0 = 1e-2;
    beta = 0.5;
    batch_size = min(D, ceil(N/16));
end
if nargin < 5
    beta = 0.5;
    batch_size = min(D, ceil(N/16));
end
if nargin < 6
    batch_size = min(D, ceil(N/16));
end

if c == 1
    [B, elapsed_time, iter] = solver.RSGM.RSGM_one(Xtilde);
else
    
    ind = randperm(N);
    group_size = ceil(N/batch_size);
    
    Y = {};
    for idx = 1:group_size
        if idx*batch_size < N
            ind_group = ind((idx-1)*batch_size + 1 : idx*batch_size);
        else
            ind_group = ind((idx-1)*batch_size + 1 : end);
        end
        Y{idx} = Xtilde(:, ind_group);
    end
    
    mu_min = 1e-15;
    if D <= N
        [U,~,~] = svd(Xtilde, 'econ');
    else
        [U,~,~] = svd(Xtilde);
    end
    B = U(:, (d+1 : end));
    
    X_sample = Y{1};
    %%% line search to determine initial step size
    BTX = B' * X_sample;
    denum = sqrt( sum( (BTX).^2 ) );
    obj_old = sum( sqrt( sum( (B' * Xtilde).^2 ) ) );

    denum(denum==0) = inf;
    X_de = X_sample ./ denum;
    grad = X_de * BTX';
    BTX_de = BTX ./ denum;
    grad = grad - B*(BTX_de * BTX');

    mu = mu_0;
    Btmp = orth(B - mu*grad);
    BtmpTX = Btmp' * X_sample;
    denum_tmp = sqrt( sum( (BtmpTX).^2 ) );
    obj_tmp = sum( denum_tmp );
    while obj_tmp / size(X_sample,2) > obj_old / N && mu > mu_min
        mu = mu*beta;
        Btmp = orth(B - mu*grad);
        BtmpTX = Btmp' * X_sample;
        denum_tmp = sqrt( sum( (BtmpTX).^2 ) );
        obj_tmp = sum( denum_tmp );
    end

    i = 0;
    while mu > mu_min && i < maxiter
        i = i+1;

        B = Btmp;

        BTX = BtmpTX;
        denum = denum_tmp;
        denum(denum==0) = inf;
        X_de = X_sample ./ denum;
        grad = X_de * BTX';
        BTX_de = BTX ./ denum;
        grad = grad - B*(BTX_de * BTX');

        mu = mu * beta;
        Btmp = orth(B - mu*grad);
        
        X_sample = Y{rem(i, group_size)+1};
        BtmpTX = Btmp' * X_sample;
        denum_tmp = sqrt( sum( (BtmpTX).^2 ) );

    end

    iter = i;
    elapsed_time = toc(t_start);
end

end