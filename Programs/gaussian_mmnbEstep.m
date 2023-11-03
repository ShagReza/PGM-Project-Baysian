function [phi,gama]=gaussian_mmnbEstep(alpha,mu,sigma,x,maskx)
%
% Author: Hanhuai Shan. 04/2012
% 
% E-step of variational EM
% 
% k: number of clusters
% d: number of features
%
% Input:
%   alpha:      k*1, parameter for Dirichlet distribution 
%   mu, sigma:  k*d, mu and sigma^2 for Gaussian distributions 
%   x:          1*d, a data point
%   maskx:      1*d, indicator vector, 1 indicates a valid entry, 0 otherwise
% 
% Output:
%    gama:      k*1, variational parameter for Dirichelet distribution
%    phi:       k*d, variational parameters for discrete distributions
%------------------------------------------------------------------------


%=================initilization==================
[k,d]=size(mu);
phi_t=ones(k,d)/k;
gama_t=alpha+d/k;

%=================iteration======================
epsilon=0.0001;
time=500;

e=100;
t=1;

temp=-(ones(k,1)*x-mu).^2./(2*sigma)-log(sqrt(2*pi*sigma));

while e>epsilon && t<time
    %--phi--
    phi_tt=exp((psi(gama_t)-psi(sum(gama_t)))*ones(1,d)+temp);
    phi_tt=(phi_tt+realmin)./(ones(k,1)*sum(phi_tt+realmin,1));
    
    %--gama--
    gama_tt=alpha+sum(phi_tt.*(ones(k,1)*maskx),2);

    %--error--
    e1=sum(sum(abs(phi_t-phi_tt)))/sum(sum(phi_t));
    e2=sum(abs(gama_t-gama_tt))/sum(gama_t);
    e=max(e1,e2);
    
    phi_t=phi_tt;
    gama_t=gama_tt;
    t=t+1;
end

gama=gama_t;
phi=phi_t;
