
%--------------------------------------------------------------------------
%Local Feature Extraction
path='E:\Coarses\ProbabilisticGraphicalModels\Project\Data\Simple\Train\300TrainFiles';
dirpath=dir([path,'\*.jpg']);
MaxDifference=0.6;
%--------------------------------------------------------------------------
n=0;
NumTrainData=length(dirpath);
LocalFeaturesCount(1:NumTrainData,1:NumCodeWords)=0;
for i=1:length(dirpath)
    i
    I=imread([path,'\',dirpath(i).name]);
    for j=6:10:251
        for k=10:10:251
            n=n+1;
            I2=I(j-5:j+5,k-5:k+5);
            I2=I2(:);
            for m=1:NumCodeWords
                x=double(I2');
                %x=mapminmax(double(I2'),0, 255);
                y=Codewords(m,:);
                dist(m)=abs(dot(x,y)/(norm(x,2)*norm(y,2))); 
            end
            [a,b]=max(dist);
%             if a<MaxDifference
%                LocalFeaturesCount(i,b)=LocalFeaturesCount(i,b)+1;
%             end
            LocalFeaturesCount(i,b)=LocalFeaturesCount(i,b)+1;
        end
    end
end
%--------------------------------------------------------------------------
trainX=LocalFeaturesCount;
trainY(1:300,1:2)=0;
trainY(1:101,1)=1;
trainY(101:200,2)=1;

LocalFeaturesCount2=[]; k=0;
for i=1:length(dirpath)
    if sum(LocalFeaturesCount(i,:))~=0
        k=k+1;
        LocalFeaturesCount2(k,:)=LocalFeaturesCount(i,:);
        trainY2(k,:)=trainY(k,:);
    end
end
trainX=LocalFeaturesCount2;
trainY=trainY2;
%--------------------------------------------------------------------------

        
