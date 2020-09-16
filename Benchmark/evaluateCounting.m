function [metsBenchmark] = evaluateCounting(seqmap, resDir, gtDataDir,benchmarkGt_name, benchmark, gt_name)

% Input:
% - seqmap
% Sequence map (e.g. `c2-train.txt` contains a list of all sequences to be 
% evaluated in a single run. These files are inside the ./seqmaps folder.
%
% - resDir
% The folder containing the tracking results. Each one should be saved in a
% separate .txt file with the name of the respective sequence (see ./res/data)
%
% - gtDataDir
% The folder containing the ground truth files.
%
% - benchmark
% The name of the benchmark, e.g. 'MOT15', 'MOT16', 'MOT17', 'DukeMTMCT'
%
% Output:
% - allMets
% Scores for each sequence
% 
% - metsBenchmark
% Aggregate score over all sequences
%
% - metsMultiCam
% Scores for multi-camera evaluation

addpath(genpath('.'));
warning off;


% Read sequence list
sequenceListFile = fullfile('seqmaps',benchmark,seqmap);

allSequences = parseSequences2(sequenceListFile);
fprintf('Sequences: \n');
disp(allSequences')
gtMat = [];
resMat = [];

% Evaluate sequences individually
allMets = [];
metsBenchmark = [];
metsMultiCam = [];

for ind = 1:length(allSequences)
    
    sequenceName = char(allSequences(ind));


    gtFilename = fullfile(gtDataDir,convertStringsToChars(benchmarkGt_name),'counter_gt',gt_name);
    gtFilename = convertStringsToChars(gtFilename);
    CounterGT = importfile_GT_Counter(gtFilename, [1, Inf]);
    CounterGT = table2array(CounterGT);
    
    sequenceFolder = [gtDataDir, convertStringsToChars(benchmarkGt_name), filesep];
    
    resFilename = [resDir, sequenceName,  '.txt'];
    resFilename = convertStringsToChars(resFilename);
    Counter_ = importfile_GT_Counter(resFilename, [1, Inf]);
    Counter_ = table2array(Counter_);

    
    sum_up_frames = 30;

    Sum_up_frame_GT = 0;
    Sum_down_frame_GT = 0;
    Sum_up_frame_Counter = 0;
    Sum_down_frame_Counter = 0;
    Error_up = 0;
    Error_down = 0;

    TP_up = 0;
    TP_down = 0;

    FN_up = 0;
    FN_down = 0;

    FP_up = 0;
    FP_down = 0;

    for i = 1:size(CounterGT,1)
    %for i = 1:479

    Sum_up_frame_GT = CounterGT(i,2) + Sum_up_frame_GT;
    Sum_down_frame_GT = CounterGT(i,3) + Sum_down_frame_GT;
    

    count_number = find(Counter_(:,1)==i);

    if ~isempty(count_number)

        Sum_up_frame_Counter = Counter_(count_number,2) + Sum_up_frame_Counter;
        Sum_down_frame_Counter = Counter_(count_number,3) + Sum_down_frame_Counter;       
   
    end
    
    if rem(i,sum_up_frames) == 0
        
        Error_up = Error_up + abs(Sum_up_frame_GT - Sum_up_frame_Counter);
        Error_down = Error_down + abs(Sum_down_frame_GT - Sum_down_frame_Counter);

        if (Sum_up_frame_GT - Sum_up_frame_Counter) > 0
            FN_up = (Sum_up_frame_GT - Sum_up_frame_Counter) + FN_up;
            TP_up = Sum_up_frame_Counter + TP_up;
        elseif (Sum_up_frame_GT - Sum_up_frame_Counter) == 0
            TP_up = Sum_up_frame_GT + TP_up;

        elseif (Sum_up_frame_GT - Sum_up_frame_Counter) < 0
            FP_up = abs(Sum_up_frame_GT - Sum_up_frame_Counter) + FP_up;
            TP_up = Sum_up_frame_GT + TP_up;
        end
       
        
        % Down Error Counter 
        
        if (Sum_down_frame_GT - Sum_down_frame_Counter) > 0
            FN_down = (Sum_down_frame_GT - Sum_down_frame_Counter) + FN_down;
            TP_down = Sum_down_frame_Counter + TP_down;
        elseif (Sum_down_frame_GT - Sum_down_frame_Counter) == 0
            TP_down = Sum_down_frame_GT + TP_down;

        elseif (Sum_down_frame_GT - Sum_down_frame_Counter) < 0
            FP_down = abs(Sum_down_frame_GT - Sum_down_frame_Counter) + FP_down;
            TP_down = Sum_down_frame_GT + TP_down;
        end
        
        
%         X = sprintf('Sum_up_frame_GT: %d',Sum_up_frame_GT);
%         disp(X)
%         X = sprintf('Sum_down_frame_GT: %d \n',Sum_down_frame_GT);
%         disp(X)
%         
%         X = sprintf('TP_up: %d \nFN_up: %d',TP_up, FN_up);
%         disp(X)
%         
%         X = sprintf('TP_down: %d \nFN_down: %d',TP_down, FN_down);
%         disp(X)
%         
%         disp("------------------")

        Sum_up_frame_GT = 0;
        Sum_down_frame_GT = 0;
        
        Sum_up_frame_Counter = 0;
        Sum_down_frame_Counter = 0;

    end
    end
    
    TP = TP_up + TP_down;
    FP = FP_up + FP_down;
    FN = FN_up + FN_down;

    counted = sum(Counter_(:,2)) + sum(Counter_(:,3));
    GT = sum(CounterGT(:,2)) + sum(CounterGT(:,3));

    Recall = TP / (TP + FN);

    Precision = TP / (TP + FP);

    F1Score = 2 * (Precision * Recall)/(Precision + Recall);
    Counted_GT = counted / GT;
    
    fprintf(' ********************* Your %s Results *********************\n', sequenceName);
    fprintf('Recall    Precision    Counted_GT   F1Score    TP    FP    FN\n');
    fprintf('%.2f      %.2f         %.2f         %.2f       %d    %d    %d \n \n', Recall, Precision, Counted_GT,F1Score,TP,FP,FN);
    
    metsBenchmark(ind,:) = [Recall, Precision, F1Score,TP, FP, FN];
    
end




