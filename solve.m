function [W] = solve(phi_x1, phi_x2, bits, param,L_tr)
[col,row] = size(phi_x1);
[~,~] = size(phi_x2);
n_anchors = param.n_anchors;
theta = param.theta;
alpha = param.alpha;
gamma = param.gamma;

B = randn(col,bits)>0;B=B*2-1;
U1 = randn(row,bits);
U2 = randn(row,bits);

C = (L_tr'*L_tr)\(L_tr'*B);
Y=L_tr;
L0 = Y;
ind1 = L0>0;
ind2 = L0 == 0;
k = inf;

threshold = 1e-4;
lastF = inf;
iter = 1;
para1=alpha/(alpha+gamma);


%% Iterative algorithm
while (iter<100)
    % update C
    J = -2*Y'*B;
    [PP,~,QQ] = svd(J,'econ');
    C = PP*QQ';
    
    % update p
    p1 = norm(B - phi_x1*U1, 'fro');
    p2 = norm(B - phi_x2*U2, 'fro');
    
    % update U1
    G1 = -2*((1/p1)*phi_x1'*B);
    [P,~,Q] = svd(G1,'econ');
    U1 = P*Q';
    
    % update U2
    G2 = -2*((1/p2)*phi_x2'*B);
    [P,~,Q] = svd(G2,'econ');
    U2 = P*Q';
    
    % update L
    Y = para1*B*C';
    Y(ind1) = e_dragging(Y(ind1),1,k);
    Y(ind2) = e_dragging(Y(ind2),-k,0);    
    
    % update B
    tt=(1/p1)*phi_x1*U1+(1/p2)*phi_x2*U2;
    B=sign(tt+alpha*Y*C);
    
    norm1 = norm(B-phi_x1*U1, 'fro') ^ 2;
    norm2 = norm(B-phi_x2*U2, 'fro') ^ 2;
    norm3 = norm(B-Y*C, 'fro') ^ 2;
    norm4 = norm(Y, 'fro') ^ 2;
    currentF = (1/p1)*norm1 +(1/p2)*norm2+ alpha*norm3 +gamma*norm4;
    if (lastF-currentF)<threshold * currentF
        if iter>3
            break
        end
    end
    iter = iter + 1;
    lastF = currentF;
end
Phi_X=[phi_x1,phi_x2];
W = (Phi_X'*Phi_X+theta*eye(2*n_anchors+2))\(Phi_X'*B);
end


