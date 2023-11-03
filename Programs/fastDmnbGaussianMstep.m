function [alpha,mu,sigma,eta]=fastDmnbGaussianMstep(alpha,eta,phi,gama,X,Y,oldSigma,maskX)
%
% variational M-step
%
% k: number of clusters.
% c: number of classes.
% d: number of features.
% n: number of data points.
%
% Input:
%   alpha:      k*1, Dirichlet parameters.
%   eta:        k*(c-1), regression parameter for c-1 classes, if
%               #class=#cluster, c=k;
%   phi:        k*n, variational parameters.
%   gama:       k*n, variational paramters.
%   X:          n*d, data matrix.
%   Y:          n*(c-1); labels for M docs with c classes, each row is a unit vector
%   oldSigma:   k*d, sigma from the last iteration.
%   maskX:      n*d, indicator matrix, 1 indicates a valid entry in X
%
% Output:
%   mu,sigma:   k*d, k*d, parameters for d features and k clusters 
%   alpha:      k*1.
%   eta:        k*(c-1)
%------------------------------------------------------------------------

[k,n]=size(phi);
[n,d]=size(X);
c=size(Y,2)+1;

% xi
xis=1+sum(phi.*(sum(exp(eta),2)*ones(1,n)),1);


% mu and sigma
for i=1:k
    s1=sum(phi(i,:)'*ones(1,d).*X.*maskX,1);
    s2=sum(phi(i,:)'*ones(1,d).*maskX,1);
    mu(i,:)=s1./s2;
    
    s3=sum(phi(i,:)'*ones(1,d).*((X-ones(n,1)*mu(i,:)).^2).*maskX,1);
    sigma(i,:)=s3./s2;
    
    
    ind=find(sigma(i,:)<10^(-30));
    if length(ind)>0
        sigma(i,ind)=oldSigma(i,ind);
%         sigma(i,ind)=10^(-30);
    end
end

% eta
s1=phi*Y;
s2=sum(phi./(ones(k,1)*xis),2);
eta=log(s1./(s2*ones(1,c-1))+realmin);


%alpha
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
    etta=1;

    alpha_tt=alpha_t-delta;
    while (length(find(alpha_tt<=0))>0)
        etta=etta/2;
        alpha_tt=alpha_t-etta*delta;
    end
    e=sum(abs(alpha_tt-alpha_t))/sum(alpha_t);
    
    alpha_t=alpha_tt;
    
    t=t+1;
end
alpha=alpha_t;