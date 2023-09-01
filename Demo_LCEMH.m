clear all;warning off;
load('I:\zcq\dataset\mir_cnn.mat')
run =5;
bits=128;

param.alpha = 1e1;
param.theta = 1e-1;
param.gamma = 1e-4;

n_anchors = 800;
param.n_anchors = n_anchors;

fprintf('centralizing data...\n');
Ntrain = size(I_tr,1);
sample = randsample(Ntrain, n_anchors);
anchorI = I_tr(sample,:);
anchorT = T_tr(sample,:);
sigmaI=100;
sigmaT=100;
PhiI = exp(-sqdist(I_tr,anchorI)/(2*sigmaI*sigmaI));
PhiI = [PhiI, ones(Ntrain,1)];
PhtT = exp(-sqdist(T_tr,anchorT)/(2*sigmaT*sigmaT));
PhtT = [PhtT, ones(Ntrain,1)];


for j = 1 : run
    
    [W] = solve(PhiI, PhtT, bits, param,L_tr);
    
    Phi_testI = exp(-sqdist(I_te,anchorI)/(2*sigmaI*sigmaI));
    Phi_testI = [Phi_testI, ones(size(Phi_testI,1),1)];
    Pht_testT = exp(-sqdist(T_te,anchorT)/(2*sigmaT*sigmaT));
    Pht_testT = [Pht_testT, ones(size(Pht_testT,1),1)];
    Phi_dbI = exp(-sqdist(I_db,anchorI)/(2*sigmaI*sigmaI));
    Phi_dbI = [Phi_dbI, ones(size(Phi_dbI,1),1)];
    Pht_dbT = exp(-sqdist(T_db,anchorT)/(2*sigmaT*sigmaT));
    Pht_dbT = [Pht_dbT, ones(size(Pht_dbT,1),1)];
    
    Phi_test = [Phi_testI,Pht_testT];
    Phi_db = [Phi_dbI,Pht_dbT];
    B_db = (Phi_db *W)>0;
    B_test = (Phi_test *W)>0;
    
    
    B_db = compactbit(B_db);
    B_test = compactbit(B_test);
    fprintf('start evaluating...\n');
    Dhamm = hammingDist(B_db, B_test);
    [P2] = perf_metric4Label( L_db, L_te, Dhamm);
    map2(j) = P2;
end
fprintf('=====================%d bits LCEMH mAP over %d iterations:%.4f==================\n', bits, run, mean(map2));
