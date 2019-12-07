clc; clear;

%% Load and prepare dataset
for i=1:4
    
    
load su

%% Prepare for launching the algorithms

% specify GO algorithm to use (BORG or NSGA2)
GOalgorithm = 'NSGA2';

% get algorithm options
global objFunOptions    

[options,objFunOptions] = ...
    getAlgorithmOptions(GOalgorithm,norm_data,true);

options.popsize=1000;
options.maxGen=100;


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
options.numObj = 4;   
options.objfun = @objFunWQEISS_regression; 

% launch
nsga2(options);     

% get solutions indexes for WQEISS
ixWQEISS = find(ix_solutions);


s={strcat('NGSA_p1000_g100_tent',num2str(i))};
save(s{1,1})
clearvars -except i
end