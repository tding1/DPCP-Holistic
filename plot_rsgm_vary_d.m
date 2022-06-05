clc
clear
close all

rng(2020);

%% settings
D = 30;
N = 500;
r = 0.7;
M = ceil(r * N / (1 - r));
d_list = [29 25 20 15 10 5];
sigma = 1e-3;  % 0  1e-6  1e-3

for i = 1:length(d_list)
    %% generate data
    d = d_list(i);
    c = D-d;
    
    barX = [randn(d,N); zeros(D-d, N)]/sqrt(d);
    O = randn(D, M)/sqrt(D); O = normc(O);
    orth_basis = [zeros(d,c); eye(c)];
    
    %% RSGM
    barE = sigma * randn(D, N) / sqrt(D);
    v_norm = vecnorm(barX+barE);
    m_norm = repmat(v_norm, D, 1);

    X_noise = (barX + barE) ./ m_norm;
    Xtilde = [X_noise, O];

    dist_list{i} = l_RSGM(Xtilde, c);

end

%% plot
plot(dist_list{1}, 'r:', 'linewidth', 2.5)
hold on
plot(dist_list{2}, 'b:', 'linewidth', 2.5)
plot(dist_list{3}, 'g:', 'linewidth', 2.5)
plot(dist_list{4}, 'r-', 'linewidth', 2)
plot(dist_list{5}, 'b-', 'linewidth', 2)
plot(dist_list{6}, 'g-', 'linewidth', 2)

legend({'$d=29$', '$d=25$', '$d=20$', '$d=15$', '$d=10$', '$d=5$'}, 'FontSize', 20, 'Interpreter','LaTex')
xlabel('iteration','FontSize', 32)
% ylabel('dist$(B,B^*)$','FontSize', 5,'FontName','Times New Roman','Interpreter','LaTex')
axis([0, 100, 1e-10, 1])
text(-22,1e-8,'re-dist$({\bf B}_t,{\bf S}^\perp)$','Rotation', 90,'FontSize',32,'FontName','Times New Roman','Interpreter','LaTex')

grid minor

set(gca, ...
    'YScale', 'log', ...
    'LineWidth' , 2                     , ...
    'XTick', 0:20:100,...
    'YTick', [1e-10 1e-5 1e-0],...
    'FontSize'  , 35             , ...
    'FontName'  , 'Times New Roman');

%% local func
function dist_list = l_RSGM(X, c)

    mu_0 = 1e-2; alpha = 1e-3; beta = 1/2;
    maxiter = 100;

    D = size(X, 1);
    d = D - c;
    A = [zeros(d,c); eye(c)];

    obj = @(B) sum(sqrt(sum((B'*X).^2,1)));

    [U,~,~] = svd(X, 'econ');
    B = U(:, (end-c+1 : end));

    %%% line search to determine initial step size
    temp = sqrt(sum((B'*X).^2,1)); indx = temp>0;
    grad = (X(:,indx)./repmat(temp(indx),D,1))*X(:,indx)'*B;
    grad = grad - B*(B'*grad);
    grad_norm = norm(grad,'fro')^2;
    eps = mu_0;
    obj_old = obj(B);
    while obj( orth(B - eps*grad) ) > obj_old - alpha*eps*grad_norm
        eps = eps*beta;
    end

    eps_o = eps;
    i = 0;
    dist_list = [];
    while i < maxiter
        i = i+1;
        temp = sqrt(sum((B'*X).^2,1)); indx = temp>0;
        grad = (X(:,indx)./repmat(temp(indx),D,1))*X(:,indx)'*B;
        grad = grad - B*(B'*grad);

        eps = eps_o*0.8^(i);
        B = orth(B - eps*grad);

        dist_list = [dist_list relative_dist(A,B)];
    end

end

