tic
close all;
clear all;
clc
rand('state', 2015)
randn('state', 2015)
%%
d= {'Dataset', 'Accuracy','Time','p','C','tau'};
 xlswrite('1111.xlsx', d,'F1');
dd2 = zeros(20,48);
pp1 = zeros(20,1);

for index=2:6
    %%
    if(index==2)
        data= xlsread('aps_failure_test_set.xlsx');
        X=data(:,1:end-1);
        Y= data(:,end);
        Y(find(Y==0))=-1;
        Ctrain=X(1:800,:);
        dtrain= Y(1:800,:);
        Ctest= X(801:end,:);
        dtest= Y(801:end,:);
        disp('aps_failure_test_set');
    end
    %%
    if (index==3)
        data= xlsread('Breast.xlsx');
        X=data(:,1:end-1);
        Y= data(:,end);
        Y(find(Y==0))=-1;
        Ctrain=X(1:250,:);
        dtrain= Y(1:250,:);
        Ctest= X(251:end,:);
        dtest= Y(251:end,:);
        disp('Breast');
    end
%     %%
    if (index==4)
        data= xlsread('creditcard.xlsx');
        X=data(:,1:end-1);
        Y= data(:,end);
        Y(find(Y==0))=-1;
        Ctrain=X(1:750,:);
        dtrain= Y(1:750,:);
        Ctest= X(751:end,:);
        dtest= Y(751:end,:);
        disp('creditcard');
    end
    %%
    if (index==5)       
        data= xlsread('diabetes_prediction.xlsx');
        X=data(:,1:end-1);
        Y= data(:,end);
        Y(find(Y==0))=-1;
        Ctrain=X(1:1150,:);
        dtrain= Y(1:1150,:);
        Ctest= X(1151:end,:);
        dtest= Y(1151:end,:);
        disp('diabetes_prediction');
    end
    
    %%
    if (index==6)
        data= xlsread('patients.xlsx');
        X=data(:,1:end-1);
        Y= data(:,end);
        Y(find(Y==0))=-1;
        Ctrain=X(1:800,:);
        dtrain= Y(1:800,:);
        Ctest= X(801:end,:);
        dtest= Y(801:end,:);
        disp('patients');
    end    

    %% Parameter setting
%      Ctrain=awgn(Ctrain,0.05); % 高斯噪声
%      Ctest=awgn(Ctest,0.05); % 高斯噪声
    Ctrain= svdatanorm(Ctrain,'svpline');
    Ctest= svdatanorm(Ctest,'scpline');
    clear C;
    kernel=1;
    C0= 2^0;
    tau=0;
    svm_tau=0;
    p1= 2^-2;
    lamb1 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
    lamb2 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
    p1val=[2^-6];
    c1val=[2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1,2^2,2^3];
%     theta = Balance_factor(Ctrain, dtrain, kernel, 0.5,p1);
%    theta = Fuzzy_MemberShip(Ctrain, dtrain, kernel, 0.5,p1);
        theta = Fuzzy_MemberShip_FCM(Ctrain, dtrain);
%         p1val=2;
%         c1val=2^-2;
%     % Prameter Tunning
%         [acc_svm, opt_p1,opt_c1,t1]= tune_para_svm(Ctrain,dtrain,Ctest,dtest,kernel,c1val,p1val);
%         p1= opt_p1;
%         pp1(index,1)= p1;
%         C0= opt_c1;
%         lamb1 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
%         lamb2 = [2^-7,2^-6,2^-5,2^-4,2^-3,2^-2,2^-1,2^0,2^1];
% %           lamb1 = 0;
% %           lamb2 = 0;
%         if (kernel==2)
%             fprintf('\n Optimal Accuracy = %3.2f with kernel parameter p1= %3.4f and C= %3.4f',acc_svm, opt_p1,opt_c1);
%         end
%         if (kernel==1)
%             fprintf('\n Optimal Accuracy = %3.2f with  C= %3.4f',acc_svm,opt_c1);
%         end
%         fprintf('\n Time Elpased  in Tunning Paramter = %3.2f seconds',t1);
%     

    %%
    [opt_tau1,opt_tau2,opt_tau3,opt_tau4,opt_tau5...
        acc_svm,acc_upsvm,acc_psvm,acc_csupsvm,acc_upldm,acc_csupldm,acc_ldm,acc_csldm,...
        time0,time1,time2,time3,time4,time5,time6,time7...
        opt_C0,opt_C1,opt_C2,opt_C3,opt_C4,opt_C5,opt_C6,opt_C7,h,...
        AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0...
        AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,...
        AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,...
        AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,...
        AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4,...
        AUC5,Sensitivity5,Specificity5,Fmeasure5,Gmeans5,...
        AUC6,Sensitivity6,Specificity6,Fmeasure6,Gmeans6,...
        AUC7,Sensitivity7,Specificity7,Fmeasure7,Gmeans7] = tune_tau(Ctrain,dtrain,Ctest,dtest,c1val,kernel,p1,theta,lamb1,lamb2);

    %% SVM
    fprintf(['\n SVM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,' ...
        'time = %3.2f,C = %3.2f'], acc_svm, AUC0,Sensitivity0,Specificity0,Fmeasure0,Gmeans0,time0,opt_C0);
    d0= [index, acc_svm,time0,p1,opt_C0,0];
    
    %% UPSVM
    fprintf('\n UPSVM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,time = %3.2f,tau = %3.2f,C = %3.2f',acc_upsvm,AUC1,Sensitivity1,Specificity1,Fmeasure1,Gmeans1,time1,opt_tau1,opt_C1);
    d1= [index, acc_upsvm,time1,p1,opt_C1,opt_tau1];
    
    %% PSVM
    fprintf('\n PSVM Accuracy=%3.2f,,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,time = %3.2f,tau = %3.2f,C = %3.2f',acc_psvm,AUC2,Sensitivity2,Specificity2,Fmeasure2,Gmeans2,time2,opt_tau2,opt_C2);
    d2= [index, acc_psvm,time2,p1,opt_C2,opt_tau2];
    
    %% CSUPSVM
    fprintf('\n CSUPSVM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,time = %3.2f,tau = %3.2f,C = %3.2f',acc_csupsvm,AUC3,Sensitivity3,Specificity3,Fmeasure3,Gmeans3,time3,opt_tau3,opt_C3);
    d3= [index, acc_csupsvm,time3,p1,opt_C3,opt_tau3];
    
    %% UPLDM
    fprintf('\n UPLDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,time = %3.2f,tau = %3.2f,C = %3.2f',acc_upldm,AUC4,Sensitivity4,Specificity4,Fmeasure4,Gmeans4,time4,opt_tau4,opt_C4);
    d4= [index, acc_upldm,time4,p1,opt_C4,opt_tau4];

    %% CSUPLDM
    fprintf('\n CSUPLDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,time = %3.2f,tau = %3.2f,C = %3.2f',acc_csupldm,AUC5,Sensitivity5,Specificity5,Fmeasure5,Gmeans5,time5,opt_tau5,opt_C5);
    d5= [index, acc_csupldm,time5,p1,opt_C5,opt_tau5];
     
    %% LDM
    fprintf('\n LDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,time = %3.2f,C = %3.2f',acc_ldm,AUC6,Sensitivity6,Specificity6,Fmeasure6,Gmeans6,time6,opt_C6);
    d6= [index, acc_ldm,time6,p1,opt_C6,0];
    
    %% CSLDM
    fprintf('\n CSLDM Accuracy=%3.2f,AUC=%3.2f,Sensitivity=%3.2f,Specificity=%3.2f,Fmeasure=%3.2f,Gmeans=%3.2f,time = %3.2f,C = %3.2f',acc_csldm,AUC7,Sensitivity7,Specificity7,Fmeasure7,Gmeans7,time7,opt_C7);
    d7= [index, acc_csldm,time7,p1,opt_C7,0];

end
    
      saveas(gcf,sprintf('Dataset%d.fig',index))
      
[m,n] = size(dd2);
A1=[];
A2=[];
A3=[];
A4=[];
A5=[];
A6=[];

A1= dd2(:,1:6);
A2= dd2(:,7:12);
A3= dd2(:,13:18);
A4= dd2(:,19:24);
A5= dd2(:,25:30);
A6= dd2(:,31:36);

final = zeros( m*6,6);
j=1;
for i=1:m
    final(j,:) = A1(i,:);
    final(j+1,:) = A2(i,:);
    final(j+2,:) = A3(i,:);
    final(j+3,:) = A4(i,:);
    final(j+4,:) = A5(i,:);
    final(j+5,:) = A6(i,:);
    j=j+6;
end

xlswrite('FCM_only.xlsx', final);
