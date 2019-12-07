%Graphcal analysis

clc

%% delta elimination
close all
delta = 5;   % The QEISS are those with accuracy at most delta% smaller than the highest one.
PFdelta_WQEISS = deltaElimination(PF_WQEISS,delta);

%% trajectories


traj_avg=nan(size(objFunOptions.Y,1),size(PFdelta_WQEISS.fvals,1));
traj_min=nan(size(objFunOptions.Y,1),size(PFdelta_WQEISS.fvals,1));
traj_max=nan(size(objFunOptions.Y,1),size(PFdelta_WQEISS.fvals,1));



for i=1:size(PFdelta_WQEISS.fvals,1) %numero subset scelti
    
    featIxes = PFdelta_WQEISS.archive(i);
    featIxes = cell2mat(featIxes);
    [~, traj_avg(:,i),traj_min(:,i), traj_max(:,i)] = trainAndValidateELM_regression(objFunOptions.PHI,objFunOptions.Y,featIxes,objFunOptions.nFolds,objFunOptions.nELM,objFunOptions.nUnits);


end

%% Plot Frequency matrices
figure('name','W-QEISS frequency matrices');
Freq=plotFrequencyMatrix(PFdelta_WQEISS,options.numVar,varNames);
Freq=Freq';
VariableChoice=[varNames(1:end-1)',num2cell(Freq)] ;


%plot trajectories
figure;
for i=1:size(PFdelta_WQEISS.fvals,1)
subplot(3,ceil(size(PFdelta_WQEISS.fvals,1)/3),i)
plot(objFunOptions.Y)
hold on
plot(traj_avg(:,i),'r')
hold on 
plot(traj_min(:,i), '-c')
plot(traj_max(:,i), '-c')
ylim([-1.5, 1.5])
xlim([0 450])
end



%plot chosen subset
figure
scatter(PFdelta_WQEISS.fvals_ext(:,4), -PFdelta_WQEISS.fvals_ext(:,3),'o');
YL=ylim;

XL= xlim;
XL(1)=XL(1)-0.1;
XL(2)=XL(2)+0.1;

xlim('manual')
axis([XL YL])

xlabel('Cardinality')
ylabel('Accuracy')


%plot obiettivi
figure;
%plot(PFdelta_WQEISS.fvals_ext(:,1),PFdelta_WQEISS.fvals_ext(:,2),PFdelta_WQEISS.fvals_ext(:,3),PFdelta_WQEISS.fvals_ext(:,4));
fvals_ext=-PFdelta_WQEISS.fvals;
m=repmat(min(fvals_ext),size(fvals_ext,1),1);
M=repmat(max(fvals_ext),size(fvals_ext,1),1);

fvals_ext=(fvals_ext-m)./(M-m);


plot([1,2,3,4],fvals_ext)
Labels={'relevance', 'redundancy', 'accuracy', 'cardinality'};
set(gca, 'XTick', 1:4, 'XTickLabel', Labels);
xlabel('objectives')
ylabel('performance to be maximised')