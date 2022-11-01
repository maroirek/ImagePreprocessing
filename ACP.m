data = [1 2 3 5 6 ; 2 3 3 7 8  ; 5 6 6 89 9 ; 7 7 8 9 8 ];

% 1- Centrage et reduction de la data
dcr = data;
for i=1:length(data(1,:))   
 dcr(:,i) = (data(:,i) - mean(data(:,i)))/max(data(:,i));   
end

% 2- matrice de covariance
covar = cov(dcr);

% 3- valeurs et vecteurs propres
[V,D] = eig(covar);
ValP = diag(D);

% 4- choix des vecteurs propres 
inertia = 0;
VecEnd=0;
LVP=length(ValP);
test=[0 0 0 0 0];
for i = 1 : 5 
    while (inertia<0.7)
        inertia = inertia + ValP(i);
        VecEnd = i ;   % index of the last vector that will be taken
    end
    test(i)= i;
end
test
V= covar(:, VecEnd: LVP);
