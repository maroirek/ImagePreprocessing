data = [1 2 3; 2 3 3; 5 6 6; 7 7 8];

% 1- Centrage et reduction de la data
dcr = data;
for i=1:length(data(1,:))   
 dcr(:,i) = (data(:,i) - mean(data(:,i)))/max(data(:,i));   
end

% 2- matrice de covariance
covar = cov(dcr);

% 3- valeurs et vecteurs propres
[V,D] = eig(covar)

