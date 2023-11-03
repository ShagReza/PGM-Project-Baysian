function [mm,perplexity]=gaussian_applyMmnb(X,maskX,alpha,mu,sigma)
%
% Author: Hanhuai Shan. 04/2012
% 
% Description:
%   Test MMNB on X
%
% k: number of clusters
% d: number of features
% n: number of data points
%
% Input:
%   alpha:      k*1
%   mu, sigma:  k*d, mu and sigma^2 for Gaussian distributions
%   X:          n*d, data matrix 
%   maskX:       n*d 0-1 matrix, 1 indicates the valid entry 
%
% Output:
%   mm:         k*n, mixed membership for each data point
%   perplexity: perplexity on X
%--------------------------------------------------------

disp(['MMNB Gaussian test...'])
[n,d]=size(X);
k=length(alpha);

for s=1:n
    x=X(s,:);
    maskx=maskX(s,:);
    [phiOne,gammaOne]=gaussian_mmnbEstep(alpha,mu,sigma,x,maskx);
    phiAll(:,:,s)=phiOne;
    gammaAll(:,s)=gammaOne;    
end

[logProb,perplexity]=gaussian_mmnbGetPerplexity(X,maskX,alpha,mu,sigma,phiAll,gammaAll);

% mm is the mixed membership on X
mm=gammaAll./(ones(k,1)*sum(gammaAll,1));
disp(['Finish test']);