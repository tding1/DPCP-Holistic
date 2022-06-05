function [b, t, iter] = RSGM_one(Xtilde,b0, mu_min,maxiter)
% Riemannian subgradient method.

t_start = tic;

if nargin < 2
    [U,~,~] = svd(Xtilde, 'econ'); 
    b0 = U(:,end);
    mu_min = 1e-15; maxiter = 200;
end

if nargin < 3
    mu_min = 1e-15; maxiter = 200;
end

if nargin < 4
    maxiter = 200;
end

obj = @(b) norm(Xtilde'*b, 1);

mu_0 = 1e-1; alpha = 1e-3; beta = 0.5;

% initialization
b = b0;

mu = mu_0;
grad = Xtilde*sign(Xtilde'*b);
grad = grad - b*(grad'*b);
grad_norm = norm(grad)^2;
% line search
tmp = b - mu*grad;
obj_old = obj(b);
while (obj(tmp/norm(tmp)) > obj_old - alpha*mu*grad_norm) && mu>mu_min
    mu = mu*beta;
    tmp = b - mu*grad;
end
b = tmp / norm(tmp);

tol = 1e-8;
Delta_J = inf;

i = 0;
while (mu > mu_min) && Delta_J > tol && i < maxiter
    i = i+1;
    grad = Xtilde*sign(Xtilde'*b);
    grad = grad - b*(grad'*b);
   
    tmp = b - mu*grad;
    b = tmp / norm(tmp);
    
    obj_new = obj(b);
    Delta_J = abs(obj_old-obj_new)/(abs(obj_old)+10^(-9));
    obj_old = obj_new;
    
    mu = mu * 0.9;
end

iter = i;

t = toc(t_start);
end