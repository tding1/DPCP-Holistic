function [B, elapsed_time] = randomized_solver(Xtilde, c, T)

t_start = tic;

if nargin < 2
   c = 1;  T = 5;
end

if nargin < 3
   T = 5 * c;
end

D = size(Xtilde, 1);

B_set = zeros(D, T);
for i = 1:T
    b0 = normc(randn(D,1));
    b = solver.RSGM.RSGM_one(Xtilde, b0);
   	B_set(:,i) = b;
end

[U, dd] = eig(B_set * B_set');
[~,ind] = sort(diag(dd), 'descend');
B = U(:,ind(1:c));

elapsed_time = toc(t_start);

end



