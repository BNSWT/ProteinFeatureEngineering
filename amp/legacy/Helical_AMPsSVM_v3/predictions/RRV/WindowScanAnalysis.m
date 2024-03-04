matlabImageDirectory = 'C:\Dropbox\Projects\Machine Learning\Helical_AMPsSVM_v3\predictions\wuhan_coronaVirus';
nameList = dir(fullfile(matlabImageDirectory));
nameList(~[nameList.isdir]) = [];
nameList = sort_nat({nameList(3:end).name}');


for gene = 1:numel(nameList);
    nameList{gene}(ismember(nameList{gene},char(65279)))=[];
end 



fileNames = fullfile(matlabImageDirectory,nameList);

for gene = 1:numel(nameList);
fileID = fopen(fullfile(fileNames{gene},'sequence.txt'));
seq = textscan(fileID,'%s');
proteinLength = numel(seq{1}{1});
fclose(fileID);

% PC
[raw] = xlsread(fullfile(fileNames{gene},'descriptors_PREDICTIONS_unsorted.csv'),'descriptors_PREDICTIONS_unsorte');
% Mac
%[raw] = readmatrix(fullfile(fileNames{gene},'descriptors_PREDICTIONS_unsorted.csv'));

seqLengths = 20:24;
firstSeq = 1;

heatMap = zeros([numel(seqLengths)+1,proteinLength]);
heatMap2 = zeros([numel(seqLengths)+1,proteinLength]);
xy_onMap = zeros([numel(seqLengths)+1,proteinLength]);

for i_length = 1:numel(seqLengths)
    
    lastSeq = firstSeq + (proteinLength-(seqLengths(i_length)-1)) - 1;
    sequencesIdx = 1:(proteinLength-(seqLengths(i_length)-1));
    
    heatMap(i_length+1,1:numel(firstSeq:lastSeq)) = raw(firstSeq:lastSeq,3);
    heatMap2(i_length+1,1:numel(firstSeq:lastSeq)) = raw(firstSeq:lastSeq,5);
    xy_onMap(i_length+1,1:numel(firstSeq:lastSeq)) = (firstSeq:lastSeq);

    firstSeq = lastSeq+1;
end 

heatMap(heatMap<=0)=nan;

figure;ax = axes;pcolor(ax,flipud(heatMap))
shading flat 
% set(ax,'YTicks',[0,10])
title(nameList{gene})

caxis(gca,[0,3])

end 


































