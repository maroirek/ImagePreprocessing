%%%%%%%%%%%%%%%%%%%%%%%%%%% ACP for matrix data  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

data = [1 2 3 5 6 ;
        2 3 3 7 8 ;
        5 6 6 89 9 ;
        7 7 8 9 8 ];

% 1- Centrage et reduction de la data
dcr = data;
for i=1:length(data(1,:))   
 dcr(:,i) = (data(:,i) - mean(data(:,i)))/std(data(:,i));   
end

% 2- matrice de covariance
covar = cov(dcr);

% 3- valeurs et vecteurs propres
[V,D] = eig(covar);
ValP = diag(D)

% 4- choix des vecteurs propres 
inertia = 0;
VecEnd=0;
LVP=length(ValP);
for i = LVP :-1 : 1 
    if (inertia<0.5)  % 0.5 = pourcentage d'info qu'on veut garder
        inertia = inertia +(ValP(i))/ sum(ValP);
        VecEnd = i ;   % index of the last vector that will be taken
    end
end

V= covar(:, VecEnd: LVP);  % matrice de passage

% 5- Data reduite en dimension

New_data= data*V;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ACP for images %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


