
% Author: Saugat Bhattacharyya
% Title: Extract the features from the ErrP data for the training data

clear all
clc

%% Import the signals
train_subs = [02,06,11,12,13,17,18,20,21,23,26]; %
for i=1:length(train_subs)
    for j=1:5
        if(train_subs(i) > 10)
        files(((i-1)*5)+j).name = strcat('train/',sprintf('Data_S%d_Sess0%d.csv',train_subs(i),j));
        else
        files(((i-1)*5)+j).name = strcat('train/',sprintf('Data_S0%d_Sess0%d.csv',train_subs(i),j));
        end
    end 
end
b = fir1(512,[0.1/100 10/100]);

tic
for i=1:length(files)
    data.raw=csvread(files(i).name,1,0);

    signal.session=data.raw(:,2:57);
    signal.time=data.raw(:,1);
    signal.feedback=data.raw(:,59);    
    
    %% Filtering the signals
    filtered=filtfilt(b,1,signal.session);
    
    smooth = sgolayfilt(filtered,3,31);
       
    %% Baseline Correction
    ind=find(signal.feedback==1);
    for j=1:length(ind)
        temp=smooth(ind(j)-40:ind(j)+199,:);
        for k=1:57
            baseline(:,k,j)=temp(41:240,k)-mean(temp(1:40,k));
        end
    end
    
    epoch = baseline(41:200,:,:);
    for j=1:size(epoch,3)
        for k=1:size(epoch,2)
            dwnsamp(:,k,j)=downsample(epoch(:,k,j),20);
        end
    end
    
    conc=reshape(dwnsamp,size(dwnsamp,1)*size(dwnsamp,2),size(dwnsamp,3));
    norm(i).data=zscore(conc');   
    clear data signal ind baseline epoch dwnsamp conc
end

feat=[];
for i=1:length(files)
    feat=vertcat(feat,norm(i).data);
end

t=toc;

train_feat=feat;

csvwrite('Across_Subject/train_feat.csv',train_feat)
