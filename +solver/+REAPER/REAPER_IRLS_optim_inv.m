function [B, t, k] = REAPER_IRLS_optim_inv(X,c,T_max,tol)

t_start = tic;

if nargin < 3
    T_max = 200;
    tol = 1e-8;
end

if nargin < 4
    tol = 1e-8;
end

[D, N] = size(X);
d = D-c;
Delta_J = Inf;
k = 0;
w = ones(1, N);
J_old = zeros(1, N);

delta = 1e-9;
while (k<T_max) && (Delta_J>tol)
    k = k + 1;
    
    BBT = REAPER_IRLS_step_optim(X,w,c);
    
    J_new = vecnorm(BBT*X);
    w = 1./max(J_new, delta);
    
    Delta_J = abs(sum(J_old)-sum(J_new))/(abs(sum(J_old))+10^(-9));
    J_old = J_new;
end

[U, ~, ~] = svd(BBT, 'econ');
B = U(:,1:c);

t = toc(t_start);

end

function BBT = REAPER_IRLS_step_optim(X,w,c)

[D,N] = size(X);
nu = zeros(D,1);
d = D - c;

lambda = zeros(D,1);
[U,S,~] = svd(sqrt(w) .* X, 'econ');
if D <= N
    lambda = diag(S);
else
    lambda(1:N) = diag(S);
end

if lambda(d+1) == 0
    nu(1:d) = ones(d,1);
else    
    for i = d+1 : D
        theta = (i-d) / sum(  1 ./ lambda(1:i)  );
        if i<D && ( lambda(i+1)==0 || ( (lambda(i)>theta) && (theta>=lambda(i+1)) ) )
            break;
        end
    end
    nu = max(0, 1-theta./lambda);
end

% P = U * diag(nu) * U';

BBT = U * diag(1-nu) * U';

end


