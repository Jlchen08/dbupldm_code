function [acc_f,AUC,Sensitivity,Specificity,Fmeasure,Gmeans, C_f] = Unified_pin_svm(X_train, Y_train, X_test,Y_test, kernel, tau, c1val,p1)
m = size(X_train,1);
H = zeros(m,m);
m1=size(X_test,1);

%% Kernel Construction
if(kernel==1)
    for i=1:m
        for j=1:m
            H(i,j) = Y_train(i)*Y_train(j)*svkernel('linear',X_train(i,:), X_train(j,:),p1);
        end
    end
end

if(kernel==2)
    for i=1:m
        for j=1:m
            H(i,j) = Y_train(i)*Y_train(j)*svkernel('rbf',X_train(i,:), X_train(j,:),p1);
        end
    end
end
acc=zeros(length(c1val),1);
auc=zeros(length(c1val),1);
sensitivity=zeros(length(c1val),1);
specificity=zeros(length(c1val),1);
fmeasure=zeros(length(c1val),1);
gmeans=zeros(length(c1val),1);
for i = 1:length(c1val)
    C0= c1val(i);
    for ii=1:size(Y_train,1)
        if (Y_train(ii)==1)
            C(ii,:)= C0;
        else
            C(ii,:)= C0*(length(find(Y_train==-1)))/(length(find(Y_train==1)));
        end
    end
    if(tau==0)
        H4 = H;
        f = -ones(m,1);
        Aeq = Y_train';
        beq= 0;
        LB = zeros(m,1);
        UB= C;
        options.Display = 'off';
        options.MaxIter = 500;
        alpha_beta = quadprog(H4, f, [], [], Aeq, beq, LB, UB, [],options);
        idx = find( (alpha_beta  > 1e-9) & ( alpha_beta  < (C-1e-19) ));
        if isempty(idx)
            b=0;
        else
            b=mean(Y_train(idx,1)-(H(idx,:)*(alpha_beta.*Y_train)));
        end
    else
                
        % Add small amount of zero order regularisation to avoid problems
        % when Hessian is badly conditioned.
        % H = H+1e-10*eye(size(H));
        
        %% Solving QPP given in eq 7
        H4 = [H, -sign(tau)*H; -sign(tau)*H, H];
        f = -[ones(m,1); -sign(tau)*ones(m,1)];
        Aeq = [Y_train', -sign(tau)*Y_train'; eye(m,m), sign(tau)*(1/tau)*eye(m,m)];
        beq= [0; C];
        LB = zeros(2*m,1);
        %%
        options.LargeScale = 'off';
        options.Display = 'off';
        options.Algorithm = 'interior-point-convex';
        lambda_beta = quadprog(H4, f, [], [], Aeq, beq, LB, [], [],options);
        alpha=lambda_beta(1:m,:);
        beta=lambda_beta(m+1:end,:);
        alpha_beta=alpha-sign(tau)*beta;
        
        %% For calculation of bias term (from eq 1 on page 4)
        idx = find( (abs(alpha )> 1e-9) & (abs(beta) > 1e-9));
        if isempty(idx)
            b=0;
        else
            b=mean(Y_train(idx,1)-(H(idx,:)*(alpha_beta.*Y_train)));
        end
        
    end
    H_test = zeros(m1, m);
    if(kernel==1)
        for ii=1:m1
            for j=1:m
                H_test(ii,j) = svkernel('linear',X_test(ii,:), X_train(j,:),p1);
            end
        end
    end
    
    if(kernel==2)
        for ii=1:m1
            for j=1:m
                H_test(ii,j) = svkernel('rbf',X_test(ii,:), X_train(j,:),p1);
            end
        end
    end
    %%
    Labels_Decision= sign(H_test*(alpha_beta.* Y_train) +b);
    Labels_Predict=Y_test;
    acc(i) = sum(Labels_Decision==Labels_Predict)*100/length(Labels_Predict);
    AA=confusionmat(Labels_Predict,Labels_Decision);
    A=rot90(AA,2);
    auc(i)= myAUC(Labels_Decision,Labels_Predict)*100;
    precision=A(1,1)/(A(1,1) + A(1,2))*100;
    sensitivity(i) = A(1,1)/(A(1,1) + A(2,1))*100;
    specificity(i) = A(2,2)/(A(1,2) + A(2,2))*100;
    fmeasure(i) = 2 * precision * sensitivity(i)/(precision + sensitivity(i));
    gmeans(i)=sqrt(sensitivity(i)*specificity(i));
    
end
[acc_f,temp] = max(acc);
[AUC,temp] = max(auc);
[Sensitivity,temp] = max(sensitivity);
[Specificity,temp] = max(specificity);
[Fmeasure,temp] = max(fmeasure);
[Gmeans,temp] = max(gmeans);
C_f = c1val(temp);
end
