
%--------------------------------------------------------------------------
path='E:\Coarses\ProbabilisticGraphicalModels\Project\Data\Simple\Train\150TrainFiles';
dirpath=dir([path,'\*.jpg']);
%--------------------------------------------------------------------------
n=0;
Patches=[];
NumCodeWords=200;
for i=1:length(dirpath)
    i
    I=imread([path,'\',dirpath(i).name]);
    for j=6:10:251
        for k=10:10:251
            n=n+1;
            I2=I(j-5:j+5,k-5:k+5);
            Patches{n,1}=I2(:);
        end
    end
end
%--------------------------------------------------------------------------
patch=[];
for i=1:n
    i
    patch(i,1:121)=Patches{i,1};
end
[idx,Codewords]  = kmeans(patch,NumCodeWords);
%--------------------------------------------------------------------------
I=[];
for i=1:NumCodeWords
    for j=1:11
        I(j,1:11)=Codewords(i,(j-1)*11+1:j*11);
    end
    subplot(20,10,i)
    imshow(uint8(I));
end
%--------------------------------------------------------------------------
