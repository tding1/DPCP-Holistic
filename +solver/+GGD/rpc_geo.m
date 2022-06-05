function [ Vk, t_end, it] = rpc_geo( X, d, maxiter, tol, mu, mu_min)
% GGD
% Note that the size of X is N-by-D


t_start = tic;

if nargin < 3
    maxiter = 200;
    tol = 1e-8;
    mu = 1e-2;
    mu_min = 1e-15;
end

if nargin < 4
    tol = 1e-8;
    mu = 1e-2;
    mu_min = 1e-15;
end

if nargin < 5
    mu = 1e-2;
    mu_min = 1e-15;
end

if nargin < 6
    mu_min = 1e-15;
end

[~,D] = size(X);

[Vk,~,~] = svd(X', 'econ');
Vk = Vk(:,1:d);
old_Cost = cost(Vk,X);

Delta_J = Inf;
i = 1;
while i < maxiter && Delta_J > tol && mu > mu_min
    
    % calculate gradient
    X_dot_vk = X * Vk;
    
    dists = vecnorm( X' - Vk * (Vk' * X') );
    dists(dists == 0 ) = inf;
    
    gradFvk = (X_dot_vk ./ dists')' * X;
    pgradFvk = gradFvk' - Vk * (Vk' * gradFvk');
    
    [U,Sigma,W] = svd(pgradFvk,'econ');
    
    %line search
    cond = 1;
    while cond
        Vkt = ( Vk*W*diag(cos(diag(Sigma*mu))) + U*diag(sin(diag(Sigma*mu))) ) *W';
        tmp = cost(Vkt,X);
        if tmp < old_Cost || mu <= mu_min
            Vk = Vkt;
            new_Cost = tmp;
            cond = 0;
        else
            mu = mu / 1.5;
        end
    end
    
    Delta_J = abs(new_Cost-old_Cost)/(abs(old_Cost)+10^(-9));
    old_Cost = new_Cost;
    
    i = i + 1;
end

it = i;
t_end = toc(t_start);

end


function [out] = cost(v,X)

out = sum(vecnorm(X' - v * (v' * X')));

end

