function [resultAlpha,resultMu,resultSigma,resultPhi,resultGama,perplexity_time]=gaussian_learnMmnb(X,maskX,oldAlpha,oldMu,oldSigma)
%
% Author: Hanhuai Shan. 04/2012
% 
% Train mmnb gaussian
%
% n: number of data points
% d: number of features
% k: number of clusters
%
% Input:
%   X:                  n*d, datamatrix
%   oldAlpha:           k*1, parameter of Dirichelt distribution
%   oldMu, oldSigma:    k*d, parameters of Poisson distributions
%   maskX:              n*d, indicator matrix, 1 indicates an valid entry
%
% Output:
%   resultAlpha:               k*1, Dirichlet distribution
%   resultMu, resultGama:      k*d, Poisson distribution
%   resultPhi:                 k*d*n, Variational Discrete distribution
%   resultGama:                k*n, Variational Dirichlet distribution
%   perplexity_time:           perplexity for each iteration
% ------------------------------------------------------------

[n,d] = size(X);
k=length(oldAlpha);

alpha_t=oldAlpha;
mu_t=oldMu;
sigma_t=oldSigma;
logProb_t=1;
perplexity_t=1;


epsilon=0.0001;
time=500;

logProb_time=[];
perplexity_time=[];

e=100;
t=1;
disp(['MMNB Gaussian training...'])
while e>epsilon && t<time
    %--E step--
    for s=1:n
        x=X(s,:);
        maskx=maskX(s,:);
        [estimatedPhi,estimatedGama]=gaussian_mmnbEstep(alpha_t,mu_t,sigma_t,x,maskx);
        phiAll(:,:,s)=estimatedPhi;
        gamaAll(:,s)=estimatedGama;    
        
    end


    [logProb_tt,perplexity_tt]=gaussian_mmnbGetPerplexity(X,maskX,alpha_t,mu_t,sigma_t,phiAll,gamaAll);
    logProb_time=[logProb_time,logProb_tt];
    perplexity_time=[perplexity_time,perplexity_tt];
    
    
    %--M step--
    [mu_tt,sigma_tt,alpha_tt]=gaussian_mmnbMstep(alpha_t,phiAll,gamaAll,X,maskX,mu_t,sigma_t);
    
    
    % ---error---
    if perplexity_tt==Inf||perplexity_t==Inf
        e=100;
    else
        e=abs(perplexity_tt-perplexity_t)/perplexity_t;
    end
    disp(['t=',int2str(t),' error= ',num2str(e), ' perplexity=',num2str(perplexity_tt)]);
    logProb_t=logProb_tt;
    perplexity_t=perplexity_tt;
   
    %---update---
    alpha_t=alpha_tt;
    mu_t=mu_tt;
    sigma_t=sigma_tt;
    
    t=t+1;
end


resultMu=mu_t;
resultSigma=sigma_t;
resultAlpha=alpha_t;
resultPhi=phiAll;
resultGama=gamaAll;




