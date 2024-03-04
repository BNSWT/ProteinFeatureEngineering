matlabImageDirectory = '/Users/yuezhang/Desktop/untitled folder/Helical_AMPsSVM_v3 copy/predictions/test_Single_WindowScan';% Change
fileID = fopen(fullfile(matlabImageDirectory,'motherSequences.txt')); % Change
seq = textscan(fileID,'%s');
proteinLength = numel(seq{1}{1});
fclose(fileID);
seqLengths = 20:24;% Change





% PC
%[raw] = xlsread(fullfile(matlabImageDirectory,'descriptors_PREDICTIONS_unsorted.csv'),'descriptors_PREDICTIONS_unsorte');
% Mac
[raw] = readmatrix(fullfile(matlabImageDirectory,'descriptors_PREDICTIONS_unsorted.csv'));

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

caxis(gca,[0,3])



































