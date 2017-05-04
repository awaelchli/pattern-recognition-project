function [ path, pathCost, matrix ] = dynamic_time_warp( features1, features2 )
%DYNAMIC_TIME_WARP Dynamic time warping (DTW) between two feature sequences
%
%   Inputs:     features1:
%               features2:
%   
%   Outputs:    path:
%               pathCost:
%               matrix:


% TODO: Warping constraints (Sakoe-Chiba band)


% Create matrix containing pairwise distances
N = size(features1, 2);
M = size(features2, 2);

matrix = inf(N, M);
for i = 1 : N
    for j = 1 : M
        
        % TODO: Decide how to compute distance with multiple types of
        % features (lower contour, upper contour, etc.)
        Fi = squeeze(features1(:, i, :));
        Fj = squeeze(features2(:, j, :));
        
        % Euclidean distance between i-th and j-th feature
        matrix(i, j) = norm(Fi - Fj);
        
    end
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
        costs(v) = matrix(next(1, v), next(2, v));
    end
    
    % Select the smallest cost 
    [ cost, direction ] = min(costs);
    
    pathCost = pathCost + cost;
    
    % Make a step in the optimal direction
    path(:, n + 1) = path(:, n) + step(:, direction);
end


end

