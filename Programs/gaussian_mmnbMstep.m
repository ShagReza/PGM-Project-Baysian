function [mu,sigma,alpha]=gaussian_mmnbMstep(alpha,phi,gama,X,maskX,oldMu,oldSigma)
%
% Author: Hanhuai Shan. 04/2012
% 
% M-step of variational EM
%
% k: number of clusters
% d: number of features
% n: number of data points
%
% Input:
%   alpha:  k*1, parameters for Dirichlet distribution
%   phi:    k*d*n, variational parameters for discrete distributions
%   gama:   k*n, variational parameters for Dirichlet distribution 
%   X:      n*d, data matrix
%   maskX:  n*d, indicator matrix, 1 indicates an valid entry
%
% Output:
%   mu, sigma:  k*d, mu and sigma^2 for Gaussian distributions
%   alpha:      k*1, Dirichlet distribution
%-----------------------------------------------------------------------

[k,N,n]=size(phi);

%===========mu and sigma===========
phi=permute(phi,[3,2,1]);
for i=1:k
   s1=sum(phi(:,:,i).*X.*maskX,1);
   s2=sum(phi(:,:,i),1);
   mu(i,:)=s1./s2;
   s3=sum(phi(:,:,i).*(X-ones(n,1)*mu(i,:)).^2.*maskX,1);
   sigma(i,:)=s3./s2;
   
   ind=find(sigma(i,:)<0.00001);
   if length(ind)>0
        disp('sigma->0');
        mu(i,ind)=oldMu(i,ind);
        sigma(i,ind)=oldSigma(i,ind);
   end
end

%============alpha================ 
alpha_t=alpha;
epsilon=0.001;
time=500;

t=0;
e=100;
psiGama=psi(gama);
psiSumGama=psi(sum(gama,1));
while e>epsilon&&t<time
    g=sum((psiGama-ones(k,1)*psiSumGama),2)+n*(psi(sum(alpha_t))-psi(alpha_t));
    h=-n*psi(1,alpha_t);
    z=n*psi(1,sum(alpha_t));
    c=sum(g./h)/(1/z+sum(1./h));
    delta=(g-c)./h;

    eta=1;
    alpha_tt=alpha_t-delta;
    while (length(find(alpha_tt<=0))>0)
        eta=eta/2;
        alpha_tt=alpha_t-eta*delta;
    end
    e=sum(abs(alpha_tt-alpha_t))/sum(alpha_t);
    
    alpha_t=alpha_tt;


    t=t+1;
end
alpha=alpha_t;