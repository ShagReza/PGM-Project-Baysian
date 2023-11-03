function [logProb, perplexity]=fastDmnbGaussianGetPerp(X,Y,alpha,mu,sigma,eta,phi,gama,maskX)
%
% Compute the perplexity
%
% k: number of components(clusters).
% c: number of classes.
% d: length of each feature vector.
% n: number of data points.
% 
% Input:
%   X:              n*d; data matrix.
%   Y:              M*(c-1); each row is the class label for one doc.  
%                   The ith dimension with value 1 indicates the doc class is
%                   i. If all dimensions are 0, the doc class is c.
%   alpha:          k*1; Dirichlet distribution
%   mu,sigma:       k*d; Gaussian distribution parameter for d features and k
%                   cluters
%   eta:            k*(c-1), regression parameter for c-1 classes, if
%                   #class=#cluster, c=k;
%   phi:            k*n, variational parameters.
%   gama:           k*n, variational paramters.
%   maskX:          n*d; 0-1 matrix with 1 indicating valide entries.
%
% Output:
%   logProb:        scaler, log-likelihood
%   perplexity:     scaler, perplexity
%-----------------------------------------------------------


[k,n]=size(phi);
[k,d]=size(mu);
ms=sum(maskX,2);

item1=n*gammaln(sum(alpha))-n*sum(gammaln(alpha))+sum((alpha-1).*sum((psi(gama)-psi(ones(k,1)*sum(gama,1))),2));


item2=sum(sum(phi.*(psi(gama)-ones(k,1)*psi(sum(gama,1))),1).*ms',2);


item3=0;
for i=1:k
    temp=sum(phi(i,:)'.*sum(maskX.*(-(X-ones(n,1)*mu(i,:)).^2./(2*ones(n,1)*sigma(i,:))-log(sqrt(2*pi*ones(n,1)*sigma(i,:)))),2));
    item3=item3+temp;
end



item4=sum(gammaln(sum(gama,1)),2)-sum(sum(gammaln(gama)))+sum(sum((gama-1).*(psi(gama)-psi(ones(k,1)*sum(gama,1)))));



item5=sum(sum(phi.*log(phi+realmin),1).*ms',2);

item6=0;
for u=1:n
    onephi=phi(:,u);
    oney=Y(u,:);
    item6=item6+onephi'*eta*oney'-log(1+sum(sum(exp(eta),2).*onephi));
end


logProb=item1+item2+item3-item4-item5+item6;
perplexity=exp(-logProb/sum(sum(maskX)));