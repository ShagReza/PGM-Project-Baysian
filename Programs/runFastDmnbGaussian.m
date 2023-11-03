clear

load data


c=3;   % 3 classes
k=3;   % 3 clusters. In this example, k=c, but we can also choose k>c

[n,d]=size(trainX);

% initialization
initalpha=rand(k,1);
initmu=[trainY,ones(n,1)-sum(trainY,2)]'*trainX;
initmu=initmu./(sum([trainY,ones(n,1)-sum(trainY,2)],1)'*ones(1,d));
initsigma=ones(k,d);
initeta=rand(k,c-1);


[alpha,mu,sigma,eta,phi,gama,logProb_time,perplexity_time]=learnFastDmnbGaussian(trainX,trainY,initalpha,initmu,initsigma,initeta,maskTrainX);

[predY,accuracy,perplexity,testphi,testgama]=applyFastDmnbGaussian(testX,testY,alpha,mu,sigma,eta,maskTestX);
