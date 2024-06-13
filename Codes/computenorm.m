%%%%%  This script computes the squared norm of v_nu - v
%% Clean
clear all, close all, clc;

%% Model parameters
xi0 = 1; eps = 0.5;  xStar = 20; c = 0.5;
mu  = 1000; vth = 0.1; kappa = 1;

%% Numerical parameters
nx  = 2^12; Lx  = 24*pi; 
nxi = 2^10; Lxi = 3;
tau = 0.0125

nt = 400; iplot = 20; ihist = iplot;

%% Spatial grid
hx  = 2*Lx/nx;  x  = -Lx +[0:nx-1]*hx;
hxi = 2*Lxi/(nxi-1); xi = -Lxi + [0:nxi-1]'*hxi;
[X,XI] = meshgrid(x,xi);

%% Function handles
wFun      = @(x) kappa*0.5*exp(-abs(x)); 
alphaFun  = @(xi,eps) (eps*sqrt(pi))^-1 * exp( -(xi-xi0).^2/eps^2)...
              .*(abs(xi - xi0) <= 2*eps);
alphaPFun  = @(xi,eps) (eps*sqrt(pi))^-1 * exp( -xi.^2/eps^2)...  
           .*(abs(xi) <= 2*eps);
SFun      = @(v) 1./(1+exp(-mu*(v-vth))); 

%% Precomputing vectors
wHat = fft(wFun(x));
alpha  = alphaFun(xi,eps);  J  = find(alpha  ~= 0); alpha  = alpha(J);
alphaP = alphaPFun(xi,eps); JP = find(alphaP ~= 0); alphaP = alphaP(JP);

%% Quadrature weights
sigma = ones(nxi,1); sigma([1 nxi]) = 0.5; sigma = hxi*sigma; sigma = sigma(JP);

%% Differentiation matrix and linear operator
e = ones(nx,1); D2 = spdiags([e -2*e e], -1:1, nxi, nxi); 
D2(1,2) = 2; D2(nxi,nxi-1) = 2; D2 = D2/hxi^2;

%%%%%%%%%%%%%%%%%%%%% change nu
for knu=1:5
    nu=0.02*knu
    A1  = (1+tau*c)*speye(nxi) - tau*nu*D2;
    dA1 = decomposition(A1);
    A0  = (1+tau*c)*speye(nxi);
    dA0 = decomposition(A0);
    %% Initial condition
    %%%%%%   V(.,.,2) -   solution with nu different from 0
   for inu=1:2
    V(:,:,inu) = ((1 - 1./(1+exp(-5*(X-xStar)))).*(X>0) + (1 - 1./(1+exp(5*(X+xStar)))).*(X<=0));
    V(:,:,inu)=V(:,:,inu).*alphaPFun(xi,eps);
    end
    t = 0;
    %% TimeStep 
    for it = 1:nt

      % Compute F
      F = zeros(nxi,nx);
      S(:,:,1)= (alphaP .* sigma)'*SFun(V(JP,:,1));
      S(:,:,2)= (alphaP .* sigma)'*SFun(V(JP,:,2));
      conv(:,:,1) = hx*ifftshift(real(ifft(fft(S(:,:,1)) .* wHat)));
      conv(:,:,2) = hx*ifftshift(real(ifft(fft(S(:,:,2)) .* wHat)));
      F(J,:,1) = alpha * conv(:,:,1);
      F(J,:,2) = alpha * conv(:,:,2);

      % Update V_1 and V_2
      V(:, :, 1) = dA0\(V(:,:,1) + tau*F(:,:,1));
      V(:, :, 2) = dA1\(V(:,:,2) + tau*F(:,:,2));
      t = t + tau
      time(it)=t;
     
    %clear dA; save('savedData.mat');
      %%%%%%%%   compute norm v_nu - v
      Vdif =V(:,:,2)-V(:,:,1);
      Vdif2= Vdif.*Vdif;
      normdif(it) = sum(sum(Vdif2))*hx*hxi;
    end
    plot(time,normdif)
    maxnorm(knu) =max(normdif);
    lnorm(knu) =sqrt(sum(normdif.^2))*tau;
end
test=maxnorm+c.*lnorm
