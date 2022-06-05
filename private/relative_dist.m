function [d] = relative_dist(A, B)

[U,~,V] = svd(A' * B);
Q = U * V';
d = norm(B-A*Q, 'fro') / sqrt(size(A,2));

end

