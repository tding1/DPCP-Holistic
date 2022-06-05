function [ Vk, t_end, it] = rpc_geo_inv_sgd( X, d, maxiter, mu, mu_min, batch_size)
% GGD
% Note that the size of X is N-by-D

t_start = tic;

[N, D] = size(X);
c = D - d;

if nargin < 3
    maxiter = 200;
    mu = 1e-2;
    mu_min = 1e-15;
    batch_size = min(D, ceil(N/16));
end

if nargin < 4
    mu = 1e-2;
    mu_min = 1e-15;
    batch_size = min(D, ceil(N/16));
end

if nargin < 5
    mu_min = 1e-15;
    batch_size = min(D, ceil(N/16));
end

if nargin < 6
   batch_size = min(D, ceil(N/16));
end

ind = randperm(N);
group_size = ceil(N/batch_size);

Y = {};
for idx = 1:group_size
    if idx*batch_size < N
        ind_group = ind((idx-1)*batch_size + 1 : idx*batch_size);
    else
        ind_group = ind((idx-1)*batch_size + 1 : end);
    end
    Y{idx} = X(ind_group, :);
end

[Vk,~,~] = svd(X', 'econ');
Vk = Vk(:,(d+1 : end));
old_Cost = cost(Vk,X);

X_sample = Y{1};
    
BTX = Vk' * X_sample';
denum = sqrt( sum( (BTX).^2 ) );

denum(denum==0) = inf;
X_de = X_sample' ./ denum;
grad = X_de * BTX';
BTX_de = BTX ./ denum;
pgradFvk = grad - Vk*(BTX_de * BTX');

[U,Sigma,W] = svd(-pgradFvk,'econ');

%line search to determine initial step size
cond = 1;
while cond
    Vkt = ( Vk*W*diag(cos(diag(Sigma*mu))) + U*diag(sin(diag(Sigma*mu))) ) *W';
    tmp = cost(Vkt,X_sample);
    if tmp / size(X_sample,1) < old_Cost / N || mu <= mu_min
        Vk = Vkt;
        cond = 0;
    else
        mu = mu / 1.5;
    end
end

i = 1;
while i < maxiter && mu > mu_min
    
    i = i+1;
    
    % calculate gradient

    X_sample = Y{rem(i, group_size)+1};
    
    BTX = Vk' * X_sample';
    denum = sqrt( sum( (BTX).^2 ) );

    denum(denum==0) = inf;
    X_de = X_sample' ./ denum;
    grad = X_de * BTX';
    BTX_de = BTX ./ denum;
    pgradFvk = grad - Vk*(BTX_de * BTX');

    
    [U,Sigma,W] = svd(-pgradFvk,'econ');
    
    if mod(i,50)==0
        mu = mu * 0.5;
    end
    Vk = ( Vk*W*diag(cos(diag(Sigma*mu))) + U*diag(sin(diag(Sigma*mu))) ) *W';
    
end

it = i;
t_end = toc(t_start);

end


function [out] = cost(v,X)

out = sum(vecnorm(v' * X'));

end

