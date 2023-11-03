function [gamma,phi]=fastMnbGaussianEstep(alpha,mu,sigma,x,maskx)
%
% Find the optimizing values of the variational parameters gamma and phi.
%
% k: number of components(clusters).
% d: length of each feature vector.
% n: number of data points.
%
% Input:
%   alpha:      k*1, Dirichlet distribution paramter. 
%   mu,sigma:   k*d matrix, each is the parameters for a specific
%               distribution of ith cluster and vth feature
%   x:          1*d, input data maxtrix. 
%   maskx:      1*d, indicator vector. 1 indicates non-missing entry in x
% 
% Output:
%   gamma:      k*1, variational parameters for to generate pi - mixing weight. 
%   phi:        k*1, variational parameters to generate z--the class.
%-----------------------------------------------------------------------

%=================initilization==================
[k,d]=size(mu);
phi_t=ones(k,1)/k;
gamma_t=alpha+d/k;
m=length(find(maskx>0));

%=================iteration======================
epsilon=0.001;
time=500;

e=100;
t=1;

temp=-(ones(k,1)*x-mu).^2./(2*sigma)-log(sqrt(2*pi*sigma));

while e>epsilon && t<time
        
    phi_tt=exp(psi(gamma_t)-psi(sum(gamma_t))+sum(temp.*(ones(k,1)*maskx),2)/m);
    
    phi_tt=(phi_tt+realmin)/sum(phi_tt+realmin); 
    
    gamma_tt=alpha+m*phi_tt;
    
    e1=sum(abs(phi_t-phi_tt))/sum(sum(phi_t));
    e2=sum(abs(gamma_t-gamma_tt))/sum(gamma_t);
    e=max(e1,e2);
    
    %c=t
    phi_t=phi_tt;
    gamma_t=gamma_tt;
    t=t+1;
end

gamma=gamma_t;
phi=phi_t;
