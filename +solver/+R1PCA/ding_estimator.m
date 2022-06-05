function [U, t_end, iter] = ding_estimator(X,d, maxiter, tol)

    t = tic;
    
    if nargin < 3
        maxiter = 200;
        tol = 1e-5;
    end
    
    if nargin < 4
        tol = 1e-5;
    end
    
    [U,~,~] = svd(X, 'econ');
    U = U(:, 1:d);
    
    sumX2 = sum(X.^2);
    UTX = U'*X;
    s = real(  sqrt(sumX2-sum(UTX.^2))  );
    c = median(s);
    
    iter=1;
    old_U = U;
    while iter < maxiter
        
        w = min(1, c./real( sqrt(sum((X-U*UTX).^2))  ) );
        U = (w .* X) * UTX';
        U = orth(U);
                
        if c == 0 || norm(U-old_U, 'fro') < tol
            U = old_U;
            break
        end
        
        old_U = U;
       
        UTX = U'*X;
        s = real(  sqrt(sumX2-sum(UTX.^2))  );
        c = median(s);

        iter = iter+1;
    end
    
    t_end = toc(t);
end
