function [B, iter, t] = RPCA_L21_OutlierDetection(Y, d, tau, lambda, budget)

tstart = tic;
% L21 robust PCA
[L, E, iter] = solver.RPCA_L21.rpca(Y, 'L21', tau, lambda, budget);
% disp(['L21 runs ', num2str(iter), ' iterations']);

[U, ~, ~] = svd(L, 'econ');
B = U(:, d+1:end);


t = toc(tstart);

end
