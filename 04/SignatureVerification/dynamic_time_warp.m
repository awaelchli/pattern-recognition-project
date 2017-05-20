function [ path, pathCost, matrix ] = dynamic_time_warp( features1, features2, r)
%DYNAMIC_TIME_WARP Dynamic time warping (DTW) between two feature sequences
%
%   Inputs:     features1:
%               features2:
%               r:
%   
%   Outputs:    path:
%               pathCost:
%               matrix:

N = size(features1, 2);
M = size(features2, 2);

% Create a mask to constrain the path along the diagonal
mask = sakoe_chiba_band(N, M, r);

% Create matrix containing pairwise distances.
% Compute distances only for the masked region, initialize all others with
% positive infinity
matrix = inf(N, M);
[ Is, Js ] = find(mask);

for k = 1 : numel(Is)
    i = Is(k);
    j = Js(k);
    
    Fi = features1(:, i);
    Fj = features2(:, j);

    % Squared euclidean distance between i-th and j-th feature
    matrix(i, j) = sum((Fi - Fj).^2);
end

% Initialize path
path = zeros(2, max(N, M));
path(:, 1) = [1, 1];

% Accumulated cost of path
pathCost = 0;

for n = 1 : size(path, 2) - 1 
    
    current_i = path(1, n); % row index
    current_j = path(2, n); % col index
    
    % Potential steps
    step = [ 1, 1, 0;       % step in i
             1, 0, 1 ];     % step in j
    
    % Remove invalid steps
    next = repmat([current_i; current_j], 1, 3) + step;
    invalid_i = next(1, :) > N;
    invalid_j = next(2, :) > M;

    invalid = invalid_i | invalid_j;
    step(invalid) = [];
    
    % Cost for each (valid) step
    costs = zeros(1, size(step, 2));
    for v = 1 : size(step, 2)
        if next(1,v)>size(matrix,1) || next(2,v)>size(matrix,2)
            fprintf("error");
        end
        costs(v) = matrix(next(1, v), next(2, v));
    end
    
    % Select the smallest cost 
    [ cost, direction ] = min(costs);
    pathCost = pathCost + cost;
    
    % Make a step in the optimal direction
    path(:, n + 1) = path(:, n) + step(:, direction);
end


end

