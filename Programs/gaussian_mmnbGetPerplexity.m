function [logProb,perplexity]=gaussian_mmnbGetPerplexity(X,maskX,alpha,mu,sigma,phi,gama)
%
% Get perplexity
%
% n: number of data points
% d: number of features
% k: number of clusters
%
% Input:
%   X:          n*d, data matrix
%   alpha:      k*1, parameter for Dirichlet distribution 
%   mu, sigma:  k*d, mu and sigma^2 for Gaussian distributions 
%   gama:       k*1, variational parameter for Dirichelet distribution
%   phi:        k*d, variational parameters for discrete distributions   
%   maskX:      n*d, indicator matrix, 1 indicates an valid entry
%
% Output:
%   logProb:    log-likelihood
%   perplexity: perplexity
%-----------------------------------------------------------------------

[k,d,n]=size(phi);

item1=n*gammaln(sum(alpha))-n*sum(gammaln(alpha))+sum((alpha-1).*sum((psi(gama)-psi(ones(k,1)*sum(gama,1))),2));

item2=0;
phi=permute(phi,[3,2,1]);
for i=1:k
    temp=phi(:,:,i).*(((psi(gama(i,:))-psi(sum(gama,1))))'*ones(1,d)).*(maskX>0);
    item2=item2+sum(sum(temp));
end

item4=sum(gammaln(sum(gama,1))-sum(gammaln(gama),1)+sum((gama-1).*(psi(gama)-ones(k,1)*psi(sum(gama,1)))));

item5=0;
for i=1:k
    temp=phi(:,:,i).*log(phi(:,:,i)+realmin).*(maskX>0);
    item5=item5+sum(sum(temp));
end

item3=0;
for i=1:k
    item3=item3+sum(sum(phi(:,:,i).*(-(X-ones(n,1)*mu(i,:)).^2./(ones(n,1)*sigma(i,:)*2)-log(sqrt(2*pi*(ones(n,1)*sigma(i,:))))).*(maskX>0)));
end

logProb=item1+item2+item3-item4-item5;
perplexity=exp(-logProb/sum(sum(maskX)));
