function [B, t, k] = REAPER_IRLS_optim_local(X,c,delta,T_max,epsilon_J,budget)

t_start = tic;

[D, N] = size(X);
d = D-c;
Delta_J = Inf;
k = 0;
w = ones(1, N);
J_old = zeros(1, N);

while (k<T_max) && (Delta_J>epsilon_J) && (toc(t_start) < budget)
    U = REAPER_IRLS_step_optim(X,w,c);
    P = U * U';
    
    J_new = 0;
    w = zeros(1, N);
    for j = 1:N
        tmp = X(:,j) - P*X(:,j);
        tmp = sqrt(sum(tmp.^2));
        J_new = J_new + tmp;
        w(j) = 1/max(tmp, delta);
    end
    clear P

%     J_new = sqrt(sum((X-P*X).^2, 1));
%     w = 1./max(J_new, delta);
    
    k = k + 1;
    Delta_J = abs(sum(J_old)-sum(J_new))/(abs(sum(J_old))+10^(-9));
    J_old = J_new;
end

[U, ~, ~] = svd(U, 'econ');
B = U(:,d+1:end);

t = toc(t_start);

clear U

end

function U = REAPER_IRLS_step_optim(X,w,c)

[D,~] = size(X);
nu = 0;
d = D - c;

[U,S,~] = svd(sqrt(w) .* X, 'econ');
lambda = diag(S);

if lambda(d+1) == 0
    nu(1:d) = ones(d,1);
else    
    for i = d+1 : size(U,2)   
        theta = (i-d) / sum(ones(i,1) ./ lambda(1:i));
        if (i<D) && (lambda(i)>theta) && (theta>=lambda(i+1))
            break;
        end
    end    
    for i = 1 : size(U,2)    
        if lambda(i)>theta
            nu(i) = 1 - (theta/lambda(i));
        else 
            nu(i) = 0;
        end
    end
end

% P = U * diag(nu) * U';
U = U * diag(sqrt(nu));

end


