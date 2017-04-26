function newFeatures = featureSelection(data,n)
%Split up the image in nxn squares and sums up the pixels inside
%data are row vector(with a square amount of pixels) of images of the mxm pixels
%newFeatures is a vector containing all nxn new features 
rowLength = size(data(1,:),2);

m=sqrt(rowLength);
stepsize = floor(m/n);
m=int8(m);
fromVector=1:stepsize:(m-mod(m,n));

toVector=stepsize:stepsize:m;
if mod(m,n)~=0
    toVector(end)=m;
end

for index=1:size(data,1)
    image=data(index,:);
    image = transpose(reshape(image,m,m));
    
    newFeature=zeros(n);
    for i=1:n
        for j=1:n
            from1=fromVector(i);
            to1=toVector(i);
            from2=fromVector(j);
            to2=toVector(j);
            square=image(from1:to1,from2:to2);
            newFeature(i,j)=sum(sum(square));
        end
    end
    newFeature=transpose(newFeature);
    newFeatures(index,:)=newFeature(:);
end





