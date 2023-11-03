
%load TrainX
%load TestX

trainX=LocalFeaturesCount;
trainY(1:300,1:2)=0;
trainY(101:200,1)=1;
trainY(201:300,2)=1;
%-----------------------------------------------------------------
            %Fast LDA:
% number of clusters
k=3;
[N,M]=size(trainX);

% initialization
alpha=rand(k,1);
beta=rand(k,M);
beta=beta./(sum(beta,2)*ones(1,M));
lap=0.0001;

% Fast LDA on training set
[resultAlpha,resultBeta,resultPhi,resultGamma,perplexity_time]=learnFastlda(trainX,alpha,beta,lap);
mm_train=resultPhi;  % resultPhi gives the membership for training docs.

% If there is a test set, run Fast LDA on test set
[mm_test,perplexity]=applyFastlda(trainX,resultAlpha,resultBeta);
%-----------------------------------------------------------------
            %Fast Discriminative LDA:
c=3;   % 3 classes
k=3;   % 3 topics. In this example, k=c, but we can also choose k>c

[M,V]=size(trainX);

initalpha=rand(k,1);
initbeta=[trainY,ones(M,1)-sum(trainY,2)]'*trainX;
initbeta=initbeta./(sum(initbeta,2)*ones(1,V));
lap=0.0001;
initeta=rand(k,c-1);

% if flag=1 use the change on perplexity to check the convergence, if flag=0, use the change on parameter to check the convergence
flag=1;
[alpha,beta,eta,phi,gama,logProb_time,perplexity_time]=learnFastDlda(trainX,trainY,initalpha,initbeta,initeta,lap,flag);
[predY,accuracy,perplexity,testphi,testgama]=applyFastDlda(trainX,trainY,alpha,beta,eta);
%-----------------------------------------------------------------
            %Fast Discriminative MNB Gaussian
c=3;   % 3 classes
k=3;   % 3 clusters. In this example, k=c, but we can also choose k>c

[n,d]=size(trainX);

% initialization
maskTrainX=ones(size(trainX));
initalpha=rand(k,1);
initmu=[trainY,ones(n,1)-sum(trainY,2)]'*trainX;
initmu=initmu./(sum([trainY,ones(n,1)-sum(trainY,2)],1)'*ones(1,d));
initsigma=ones(k,d);
initeta=rand(k,c-1);

[alpha,mu,sigma,eta,phi,gama,logProb_time,perplexity_time]=learnFastDmnbGaussian(trainX,trainY,initalpha,initmu,initsigma,initeta,maskTrainX);
[predY,accuracy,perplexity,testphi,testgama]=applyFastDmnbGaussian(trainX,trainY,alpha,mu,sigma,eta,maskTrainX);
%-----------------------------------------------------------------

