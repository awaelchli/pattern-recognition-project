%% Pattern Recognition Project 4
clear all;
close all;

%% Load enrollment
fnames = dir('enrollment/*.txt');
numfids = length(fnames);
enrollment = cell(1,numfids);
for K = 1:numfids
  enrollment{K} = load(['enrollment/',fnames(K).name]);
end

%% Load verification
load('gt.mat');
fnames = dir('verification/*.txt');
numfids = length(fnames);
verification = cell(1,numfids);
for K = 1:numfids
  verification{K} = load(['verification/',fnames(K).name]);
end

clear fnames;
clear K;
clear numfids;

%% Calculate Velocity
for i=1:length(enrollment)
    vx = zeros(size(enrollment{i},1),1);
    vy = zeros(size(enrollment{i},1),1);
    vx(1) = 0;
    vy(1) = 0;
    for u=2:size(enrollment{i},1)
        dt = (enrollment{i}(u,1) - enrollment{i}(u-1,1));
        dx = (enrollment{i}(u,2) - enrollment{i}(u-1,2));
        dy = (enrollment{i}(u,3) - enrollment{i}(u-1,3));
        vx(u) = dx/dt;
        vy(u) = dy/dt;
    end
    enrollment{i} = [enrollment{i}, vx, vy];
end

for i=1:length(verification)
    vx = zeros(size(verification{i},1),1);
    vy = zeros(size(verification{i},1),1);
    vx(1) = 0;
    vy(1) = 0;
    for u=2:size(verification{i},1)
        dt = (verification{i}(u,1) - verification{i}(u-1,1));
        dx = (verification{i}(u,2) - verification{i}(u-1,2));
        dy = (verification{i}(u,3) - verification{i}(u-1,3));
        vx(u) = dx/dt;
        vy(u) = dy/dt;
    end
    verification{i} = [verification{i}, vx, vy];
end

clear vx;
clear vy;
clear dt;
clear dx;
clear dy;

%% Normalize
for i=1:length(enrollment)
   for u=1:size(enrollment{1},2)
       enrollment{i}(:,u) = normalize(enrollment{i}(:,u));
   end
end
for i=1:length(verification)
   for u=1:size(verification{1},2)
       verification{i}(:,u) = normalize(verification{i}(:,u));
   end
end

clear i;
clear u;

%% Verification
numberOfUsers = 30;
numberOfEnrollments = 5;
numberOfSignaturesPerUser = 45;

classification = inf;
fileID = fopen('results.txt','w');
for k=1:numberOfUsers 
    tmp_classification = false(numberOfSignaturesPerUser,1);
    tmp_cost = zeros(numberOfEnrollments,1);
    tmp_distance = zeros(numberOfSignaturesPerUser,1);
    for i=1:numberOfSignaturesPerUser 
        fprintf("Signature: %d\n",(k-1)*numberOfSignaturesPerUser+i);
        for u=1:numberOfEnrollments
            %[~, tmp_cost(u), ~] = dynamic_time_warp(verification{(k-1)*numberOfSignaturesPerUser+i}(:,[2,3,4,8,9])',enrollment{(k-1)*numberOfEnrollments+u}(:,[2,3,4,8,9])',20);
            tmp_cost(u)=dtw(verification{(k-1)*numberOfSignaturesPerUser+i}(:,[2,3,4,8,9]),enrollment{(k-1)*numberOfEnrollments+u}(:,[2,3,4,8,9]),4);
        end
        tmp_distance(i) = mean(tmp_cost);
        fprintf("distance: %d\n",mean(tmp_distance(i)));
    end
    output = sprintf("%03d",k);
    [A,I] = sort(tmp_distance);
    for t=1:length(A)
        output = sprintf("%s %02d,%f",output,I(t),A(t));
    end
    fprintf(fileID,"%s",output);
    fprintf(fileID,"\n");
    tmp_classification(I(1:20,1))=true;
    classification = [classification; tmp_classification];
end
fclose(fileID);
groundTruth = false(numberOfSignaturesPerUser*numberOfUsers,1);
groundTruth(gt.g(1:numberOfSignaturesPerUser*numberOfUsers)=='g') = true;
classification = classification(2:end);
accuracy = sum(groundTruth(1:numberOfSignaturesPerUser*numberOfUsers) == classification)/length(classification);
sprintf("Accuracy: %f", accuracy)
precision = sum(classification(groundTruth(1:numberOfSignaturesPerUser*numberOfUsers)==true))/sum(classification);
sprintf("Precision: %f", precision)
clear i;
clear u;

%% Plot
id = 6;
figure;
subplot(6,1,[1,2,3])
plot(enrollment{id}(:,2),enrollment{id}(:,3),'LineWidth',2)
ylabel('Signature')
subplot(6,1,4)
plot(enrollment{id}(:,1),enrollment{id}(:,4),'LineWidth',2)
ylabel('Pressure')
subplot(6,1,5)
plot(enrollment{id}(:,1),enrollment{id}(:,8),'LineWidth',2)
ylabel('vx')
subplot(6,1,6)
plot(enrollment{id}(:,1),enrollment{id}(:,9),'LineWidth',2)
ylabel('vy')

clear id;
