clc
clear

rng('default')
rng(2020);

D = 100;
c = ceil(0.05*D);
d = D - c;
N = 50 * D;

r = 0.5;
M = ceil(r * N / (1 - r));
sigma = 0.05;

barX = [randn(d,N); zeros(D-d, N)]/sqrt(d);
O = randn(D, M)/sqrt(D); O = normc(O);
A = [zeros(d,c); eye(c)];

barE = sigma * randn(D, N) / sqrt(D);
v_norm = vecnorm(barX+barE);
m_norm = repmat(v_norm, D, 1);

X_noise = (barX + barE) ./ m_norm;
Xtilde = [X_noise, O];

%% PCA
tic;
[U,~,~] = svd(Xtilde, 'econ');
B = U(:, (d+1 : end));
t_pca = toc;
dist_pca = relative_dist(A, B);

%% R1PCA
[P, t_r1pca, it_r1pca] = solver.R1PCA.ding_estimator(Xtilde, d, 200, 1e-5);
dist_r1pca = relative_dist(A, null(P'));

%% REAPER
[P, t_reaper, it_reaper] = solver.REAPER.REAPER_IRLS_optim(Xtilde,c,200,1e-8);
dist_reaper = relative_dist(A, null(P'));

%% GGD
[P, t_ggd, it_ggd] = solver.GGD.rpc_geo(Xtilde', d, 200, 1e-8, 1e-1, 1e-15);
dist_ggd = relative_dist(A, null(P'));

%% GGD-dual
[B, t_ggd_dual, it_ggd_dual] = solver.GGD.rpc_geo_inv(Xtilde', d, 200, 1e-8, 1e-1, 1e-15);
dist_ggd_dual = relative_dist(A, B);

%% Recursive approach
[B,t_recursive] = solver.RSGM.recursive_solver(Xtilde, c);
dist_recursive = relative_dist(A, B);

%% Our apporach
[B, t_holistic, it_rsgm] = solver.RSGM.RSGM_entire(Xtilde, c, 200, 1e-8, .1, .9);
dist_holistic = relative_dist(A, B);

%% report
fprintf('       dist_pca: %.4f,        t_pca: %.4f\n',dist_pca, t_pca)
fprintf('     dist_r1pca: %.4f,      t_r1pca: %.4f\n',dist_r1pca, t_r1pca)
fprintf('    dist_reaper: %.4f,     t_reaper: %.4f\n',dist_reaper, t_reaper)
fprintf('       dist_GGD: %.4f,        t_GGD: %.4f\n',dist_ggd, t_ggd)
fprintf('  dist_GGD_dual: %.4f,   t_GGD_dual: %.4f\n',dist_ggd_dual, t_ggd_dual)
fprintf(' dist_recursive: %.4f,  t_recursive: %.4f\n',dist_recursive, t_recursive)
fprintf('  dist_holistic: %.4f,   t_holistic: %.4f\n',dist_holistic, t_holistic)








