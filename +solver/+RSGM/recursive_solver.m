function [B, elapsed_time] = recursive_solver(Xtilde, c)

t_start = tic;

if nargin < 2
   c = 1;
end

B = [];
for i = 1:c
    if numel(B) == 0
        b = solver.RSGM.RSGM_one(Xtilde);
    else
        A_orth = null(B');
        Y = A_orth' * Xtilde;
        b = A_orth * solver.RSGM.RSGM_one(Y);
    end
    B = [B, b];
end

elapsed_time = toc(t_start);

end