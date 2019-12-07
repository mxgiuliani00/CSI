clear
clc

%% load data (this example focus on the seasonal prediction of Jan-Feb-Mar precipitation based on Oct-Nov-Dec SST)
% output of 1st step of CSI (i.e. NIPA teleconnection analysis): first PC aggregating the selected SSTs for each phase of ENSO and NAO, along with climate state flag defined by the combination of ENSO-NAO phases
load -ascii pc_mei_ja.txt           % 1st PC for ENSO
load -ascii pc_nao_ja.txt           % 1st PC for NAO
load -ascii type_meinao_ja.txt      % climate-state flag
% observed seasonal precipitation
load -ascii rain_JFM.txt

%% prepare datasets
t   = type_meinao_ja;
p1  = pc_mei_ja;
p2  = pc_nao_ja;
r   = rain_JFM;

PHI = [p1,p2,t] ;
g   = [p1,p2,t,r];
Y=r;

%% Extreme Learning Machine model with leave-one-out crossvalidation
nELM=10;
nUnits=10;
nFolds=38;
featIxes=[1,2,3];
Yhat = zeros(size(Y,1),1);
SU = zeros(1,nELM) + Inf;

Rmax=0;
maxSU=-inf;
k=1;

for j = 1 : nELM
    
    % k-fold cross validation
    lData  = size(Y,1);
    lFold  = floor(lData/nFolds);
    
    for i = 1 : nFolds
        % select trainind and validation data
        ix1 = (i-1)*lFold+1;
        if i == nFolds
            ix2 = lData;
        else
            ix2 = i*lFold;
        end
        valIxes  = ix1:ix2; % select the validation chunk
        trIxes = setdiff(1:lData,valIxes); % obtain training indexes by set difference
        valIxes=trIxes;
        
        % create datasets
        trX  = PHI(trIxes,featIxes);  trY  = Y(trIxes,:);
        valX = PHI(valIxes,featIxes);
        valX =trX;
        
        % train and test ELM
        [~,Yhat(valIxes)] =...
            ELMregression(trX', trY', valX', nUnits);
    end
    Traj(:,j)=Yhat; 
    SU(j) = computeSU(Y,Yhat);
    R2(j)=rsq(Y,Yhat);
    
    if R2(j)>Rmax   %SU(j)>maxSU
        maxSU=SU;
        k=j;
        Rmax=R2(j);
    end
    
    
end
pred_r=Traj(:,k);
C = corrcoef(pred_r,r)

%% results
figure; plot(r,pred_r, '.'); 
xlabel('observed seasonal precipitation [mm]');
ylabel('predicted seasonal precipitation [mm]');

year=[1971:1:2008];
figure; plot (year,r, 'k-');
hold on; plot (year, pred_r, 'r-');
legend('observed', 'predicted');
ylabel('seasonal precipitation [mm]')

