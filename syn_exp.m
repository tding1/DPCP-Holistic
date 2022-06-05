clc
clear

rng('default')
rng(2021);

D = 100;
c = ceil(0.05*D);
d = D - c;
N = 10 * D;

methods = {'pca', 'r1pca', 'holistic', 'recursive', 'reaper', 'ggd', 'ggd_dual'};
method = methods{3};

sigma_list = 0:0.05:0.5;
r_list = 0.1:0.1:0.7;
data = zeros(length(sigma_list), length(r_list));
time = zeros(length(sigma_list), length(r_list));

T = 10;

i = 1;
for sigma = sigma_list
    j = 1;
    for r = r_list
        M = ceil(r * N / (1 - r));
        A = [zeros(d,c); eye(c)];
        
        dist = 0;
        tm = 0;
        for repeat = 1:T
            barX = [randn(d,N); zeros(D-d, N)]/sqrt(d);
            O = randn(D, M)/sqrt(D); O = normc(O);

            barE = sigma * randn(D, N) / sqrt(D);
            v_norm = vecnorm(barX+barE);
            m_norm = repmat(v_norm, D, 1);

            X_noise = (barX + barE) ./ m_norm;
            Xtilde = [X_noise, O];
            
            if strcmp(method, 'pca')
                tic;
                [U,~,~] = svd(Xtilde, 'econ');
                B = U(:, (d+1 : end));
                t_pca = toc;
                dist = dist + relative_dist(A, B);
                tm = tm + t_pca;
            end
            
            if strcmp(method, 'r1pca')
                [P, t_r1pca, ~] = solver.R1PCA.ding_estimator(Xtilde, d, 200, 1e-5);
                dist = dist + relative_dist(A, null(P'));
                tm = tm + t_r1pca;
            end
            
            if strcmp(method, 'holistic')
                [B, t_rsgm, ~] = solver.RSGM.RSGM_entire(Xtilde, c, 200, 1e-8, .1, .9);
                dist = dist + relative_dist(A, B);
                tm = tm + t_rsgm;
            end

            if strcmp(method, 'recursive')
                [B,t_recursive] = solver.RSGM.recursive_solver(Xtilde, c);
                dist = dist + relative_dist(A, B);
                tm = tm + t_recursive;
            end

            if strcmp(method, 'reaper')
                [P, t_reaper, ~] = solver.REAPER.REAPER_IRLS_optim(Xtilde,c,200,1e-8);
                dist = dist + relative_dist(A, null(P'));
                tm = tm + t_reaper;
            end
            
            if strcmp(method, 'ggd')
                [P, t_ggd, ~] = solver.GGD.rpc_geo(Xtilde', d, 200, 1e-8, 1e-1, 1e-15);
                dist = dist + relative_dist(A, null(P'));
                tm = tm + t_ggd;
            end
            
            if strcmp(method, 'ggd_dual')
                [B, t_ggd_dual, ~] = solver.GGD.rpc_geo_inv(Xtilde', d, 200, 1e-8, 1e-1, 1e-15);
                dist = dist + relative_dist(A, B);
                tm = tm + t_ggd_dual;
            end
            
        end
        
        data(end-i+1, j) = dist / T;
        time(end-i+1, j) = tm / T;
        
        j = j + 1;
    end
    i = i + 1;
end

hf = figure('visible','off');

imagesc(r_list,sigma_list,1-flipud(data)); colormap(gray(256));  caxis([0 1]);
xlabel('$M/(M+N)$','FontSize',30,'FontName','Times New Roman','Interpreter','LaTex');

set(gca,'YDir','normal')
set(gca, ...
    'LineWidth' , 2                     , ...
    'FontSize'  , 30              , ...
    'FontName'  , 'Times New Roman'     , ...
    'YTick', sigma_list,...
    'XTick', r_list           );
    xtickformat('%.1f')
    ytickformat('%.2f')
   text(-0.1,0.25,'$\sigma$','FontSize',35,'FontName','Times New Roman','Interpreter','LaTex')

set(gcf, 'Color', 'white');

save_dir = 'syn_results/';
if ~isfolder(save_dir)
    mkdir(save_dir);
end

file_name = ['D_' num2str(D) '_c_' num2str(c) '_N_' num2str(N) '_' method];

print(hf,'-dpdf', '-bestfit', [save_dir file_name '.pdf']);
save([save_dir file_name '.mat'], 'data', 'time');



