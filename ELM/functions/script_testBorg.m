clear
clc

lb = zeros(1,11);
ub = ones(1,11);
e = 0.01*ones(1,2);
NFE = 10^6;

vv = [];
JJ = [];
for i = 1:10
    [v, J]=borg( 11,2,0,@DTLZ2, NFE, lb, ub, e );
    vv = [vv; v];
    JJ = [JJ; J];
end

save workspace.mat