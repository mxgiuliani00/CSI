function [val, idx_siso] = iis_nemo(subset,M,nmin,ns,p,max_iter,Vflag)

% This function implements the IIS algorithm
%
% subset   = observations
% M        = number of trees in the ensemble
% nmin     = minimum number of points per leaf 
% ns       = number of folds in the k-fold cross-validation process 
% p        = number of SISO models (it must be smaller than the number of
%            candidate inputs).
% epsilon  = tolerance
% max_iter = maximum number of iterations
% Vflag     = selection of the type of validation, 
%               1 = k-fold(default)
%               2= repeated random sub-sampling
%
% Output
% result   = structure containing the result for each iteration
%
%
% Copyright 2014 Stefano Galelli and Matteo Giuliani
% Assistant Professor, Singapore University of Technology and Design
% stefano_galelli@sutd.edu.sg
% http://people.sutd.edu.sg/~stefano_galelli/index.html
% Research Fellow, Politecnico di Milano
% matteo.giuliani@polimi.it
% http://giuliani.faculty.polimi.it
%
% Please refer to README.txt for further information.
%
%
% This file is part of MATLAB_IterativeInputSelection.
% 
%     MATLAB_IterativeInputSelection is free software: you can redistribute 
%     it and/or modify it under the terms of the GNU General Public License 
%     as published by the Free Software Foundation, either version 3 of the 
%     License, or (at your option) any later version.     
% 
%     This code is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with MATLAB_IterativeInputSelection.  
%     If not, see <http://www.gnu.org/licenses/>.
% 

val= nan(max_iter, 1); 
idx_siso = nan(max_iter, 1);

if nargin<8
    f = 1 ;
else
    f = Vflag ; 
end

% 0) SET THE PARAMETERS

% Initialize the counter and the exit condition flag
iter     = 1;    % iterations counter
diff     = 1;    % exit flag 

% Re-define the subset matrix
l = floor(length(subset)/ns);
subset = subset(1:l*ns,:);

% Define the MISO model output
miso_output = subset(:,end);

% Define the set of candidate input variables
input  = subset(:,1:end-1);

% Other variables to be initialized      
miso_input = [];  % initialize an empty set to store the input variables to be selected%

    for iter = 1:max_iter
    % Visualize the iteration
%    disp('ITERATION:'); disp(iter);
    
    % Define the output variable to be used during the ranking
    if iter == 1 
        rank_output = miso_output; % at the first iteration the MISO model output and ranking output are the same variable%
    else         
        rank_output = residual;    % at the other iterations, the ranking output is the residual of the previous MISO model%
    end
    
    % Define the ranking matrix
    matrix_ranking = [input rank_output];
    
    % Run the feature ranking
%    disp('Ranking:');
    k = size(input,2);
    %[ranking] = input_ranking(matrix_ranking,M,k,nmin);     

    % Select and cross-validate p SISO models (the first p-ranked models)
 %   disp('Evaluating SISO models:');
    features = ranking(1:p,2);                             % p features to be considered           
    performance = zeros(p,1);	                           % initialize a vector for the performance of the p SISO models%			     
    nELM = 30;
    nUnits = 5;
    for i = 1:p

        SU = trainAndValidateELM_regression(input,miso_output,features(i),ns,nELM,nUnits);

		performance(i) = SU;%siso_model.cross_validation.performance.Rt2_val_pred_mean;
     end

    
    % Choose the SISO model with the best performance
	[val(iter),idx_siso(iter)] = max(performance);
    end




