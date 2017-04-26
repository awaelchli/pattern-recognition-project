%Cs
% x = linspace(-5, 15, 10);
% y = [19.04817, 36.78521, 47.93334, 60.53334, 64.94816, 65.56668, 65.62964, 65.50372, 65.39261, 65.38149];
% plot(x,y,'x','MarkerIndices', 1:1:length(y));
% title('Average accuracy for different Cs')
% ylabel('accuracy')
% xlabel('C = 2^x')

%gammas
% x = linspace(-15, 3, 10);
% y = [62.19631, 69.23704, 76.32221,83.54445, 90.52223, 87.76668, 47.37039, 17.02225, 11.3704, 11.3704];
% plot(x,y,'x','MarkerIndices', 1:1:length(y));
% title('Average accuracy for different gammas')
% ylabel('accuracy')
% xlabel('gamma = 2^x')

%Cs (linear kernel)
x = linspace(-5, 15, 10);
y = [91.6296, 90.9259, 90.0741, 90.037, 90.037, 90.037, 90.037, 90.037, 90.037, 90.037];
plot(x,y,'x','MarkerIndices', 1:1:length(y));
title('Average accuracy for different Cs (linear kernel)')
ylabel('accuracy')
xlabel('C = 2^x')


