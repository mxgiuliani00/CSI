% load dataset


for i=1:5

load DefSM_perBorg
%% Prepare for launching the algorithms

% specify GO algorithm to use (BORG or NSGA2)
GOalgorithm = 'BORG';

% get algorithm options
global objFunOptions    

[options,objFunOptions] = ...
    getAlgorithmOptions(GOalgorithm,norm_data,true);



options.NFE=50000;




% initialize overall archive and array containing the values of the
% objctive functions (fvals)
global archive fvals ix_solutions
archive = {};               % archive of all solutions explored
fvals   = [];               % values of the obj function explored
                            %   RELEVANCE - REDUNDACY - SU - #INPUTS  

ix_solutions = [];          % this will track which solutions are found by each algorithm


%% launch WQEISS
fprintf ('Launching WQEISS\n')

% define number of obj functions and the matlab function coding them
options.nobjs = 4;   
options.objectiveFcn = @objFunWQEISS_regression; 
epsilon = 10^-3;
epsilons = repmat(epsilon, [1,options.nobjs]);

% launch
borg(...
    options.nvars,options.nobjs,options.nconstrs,...
    options.objectiveFcn, options.NFE,...
    options.lowerBounds, options.upperBounds, epsilons);


% get solutions indexes for WQEISS
ixWQEISS = find(ix_solutions);

s={strcat('BORG_defSM_50 000_tent',num2str(i))};
save(s{1,1})
clearvars -except i

end