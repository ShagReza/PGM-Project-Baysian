
% Author: Hanhuai Shan. 04/2012

clear;

load iris;
[m,n]=size(data);
trainX=data(randpermn(1:round(m*0.8)),:);
trainMask=ones(size(trainX));
testX=data(randpermn(round(m*0.8)+1:end),:);
testMask=ones(size(testX));

% initialization
k=3;
alpha=ones(k,1)/k;
mu(1,:)=data(1,:);
mu(2,:)=data(51,:);
mu(3,:)=data(101,:);
sigma=ones(k,n);

% training
[resultAlpha,resultMu,resultSigma,resultPhi,resultGama,perplexity_time]=gaussian_learnMmnb(trainX,trainMask,alpha,mu,sigma);
mm_train=resultGama./(ones(k,1)*sum(resultGama,1)); % mixed membership on the training set

% test
[mm_test,perplexity]=gaussian_applyMmnb(testX,testMask,resultAlpha,resultMu,resultSigma); 
