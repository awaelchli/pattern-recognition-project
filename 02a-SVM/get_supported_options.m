function [ options ] = get_supported_options( ~ )
%GET_SUPPORTED_OPTIONS Returns a struct of supported options
%
%   To build a custom option set for SVM, one can simply choose from this
%   struct and concatenate the string.

% Type of SVM
options.type.c_svc = '-s 0 ';
options.type.nu_svc = '-s 1 ';
options.type.one_class_svm = '-s 2 ';
options.type.epsilon_svr = '-s 3 ';
options.type.nu_svr = '-s 4 ';

% Supported kernels
options.kernel.linear = '-t 0 ';
options.kernel.polynomial = '-t 1 ';
options.kernel.rbf = '-t 2 ';
options.kernel.sigmoid = '-t 3 ';

% Additional kernel options
options.kernel.degree = @(deg) sprintf('-d %d ', deg);
options.kernel.gamma = @(gamma) sprintf('-g %d ', gamma);
options.kernel.coeff0 = @(coeff0) sprintf('-r %d ', coeff0);

% Other options
options.cost = @(C) sprintf('-c %d ', C);
options.nu = @(nu) sprintf('-n %d ', nu);
options.cachesize = @(mb) sprintf('-m %d ', mb);
options.cross_validation = @(n) sprintf('-v %d ', n);
options.quiet = '-q ';

end

