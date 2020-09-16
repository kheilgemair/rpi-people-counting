function [metsBenchmark] = evaluateDetection_1(seqmap, resDir, gtDataDir,benchmarkGt_name, benchmark, gt_name)

addpath(genpath('.'));
warning off;

display = 0;
% Read sequence list
sequenceListFile = fullfile('seqmaps',seqmap);

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
    
    
%     switch int8(ind)
%         case 1
%             gt_name = "gt600";
%         case 2
%             gt_name = "gt800";
%         case 3
%             gt_name = "gt1200";
%         case 4
%             gt_name = "gt1920";
%     end
   
    gtFilename = fullfile(gtDataDir,convertStringsToChars(benchmarkGt_name),'gt',gt_name);
    gtFilename = convertStringsToChars(gtFilename);
    
    url_gt = gtFilename;
    
    sequenceFolder = [gtDataDir, convertStringsToChars(benchmarkGt_name), filesep];
    
    resFilename = [resDir, sequenceName,  '.txt'];
    resFilename = convertStringsToChars(resFilename);

    disp(gt_name)
    disp(sequenceName)
    url_det = resFilename;


    %% Setup the Import Options
    opts = delimitedTextImportOptions("NumVariables", 10);

    % Specify range and delimiter
    opts.DataLines = [1, Inf];
    opts.Delimiter = ",";

    % Specify column names and types
    opts.VariableNames = ["VarName1", "VarName2", "VarName3", "VarName4", "VarName5", "VarName6", "VarName7", "VarName8", "VarName9", "VarName10"];
    opts.VariableTypes = ["double", "double", "double", "double", "double", "double", "double", "double", "double", "double"];
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";

    %% Import the data
    gt = readtable(url_gt, opts);
    det =  readtable(url_det, opts);


    gt = table2array(gt);
    det = table2array(det);


    %det = Sort;

    clear url_gt url_det url_Sort opts


    for counter = 1:1500     

        gt_boxes = find(gt(:,1)==counter);
        det_boxes = find(det(:,1)==counter);

        groundTruthBboxes = gt(gt_boxes,3:6);
        bboxes = det(det_boxes,3:6);

        [precision,recall] = bboxPrecisionRecall(bboxes,groundTruthBboxes,0.5);

        if display == 1
            frame = readFrame(v);
            image(frame)
            hold on

            % Red Bounding Boxes
            for i=1:size(groundTruthBboxes,1)
                rectangle('Position',groundTruthBboxes(i,:),'EdgeColor','r');
            end

            % Blue bounding boxes
            for i=1:size(bboxes,1)
                rectangle('Position',bboxes(i,:),'EdgeColor','b');
            end 
        end


        %% Calculate Performance

        GT_Num_BB(counter,1) = size(gt_boxes,1);
        Det_Num_BB = size(det_boxes,1);


        Correct_Det = recall * GT_Num_BB(counter,1);

        if counter == 1500
           disp('Stop');
           break
        end 

        precision_mat(counter,1) = precision;
        recall_mat(counter,1) = recall;
        TP(counter,1) = Correct_Det;
        if Correct_Det==0 && precision==0 && Correct_Det == 0
            FP(counter,1) = 0;
        else
            FP(counter,1) = (Correct_Det / precision) - Correct_Det;
        end

        FN(counter,1) = GT_Num_BB(counter,1) - TP(counter,1);

        if display == 1
            disp("-----------------------")
            disp("TP")
            disp(TP(counter,1))
            disp("FP")
            disp(FP(counter,1))
            disp("FN")
            disp(FN(counter,1))

            if display == 1
                pause(0.1)
            end
        end
    end


    %% Calc overall Performance 

    TP_all = sum(TP);
    FP_all = sum(FP);
    FN_all = sum(FN);

    GT_all = sum(GT_Num_BB);

   % disp("-----------------------")

    Precision_all = TP_all / (TP_all + FP_all);
   % disp("Precision")
   % disp(Precision_all)

    Recall_all = TP_all  / GT_all;
    %disp("Recall")
    %disp(Recall_all)

    AFA = FP_all / counter;
    %disp("AFA")
    %disp(AFA)

    AMT = FN_all / counter;
    %disp("AMT")
    %disp(AMT)

    %disp("TP")
    %disp(TP_all)

    %disp("FP")
    %disp(FP_all)

    %disp("FN")
    %disp(FN_all)
    

    


    %ID_gt = unique(gt(:,2));


    fprintf(' ********************* Your %s Results *********************\n', sequenceName);
    fprintf('Recall    Precision    AFA    AMT      TP         FP         FN    \n');
    fprintf('%.2f      %.2f         %.2f   %.2f     %d      %d      %d \n \n', Recall_all, Precision_all, AFA,AMT, TP_all,FP_all,FN_all);

    metsBenchmark(ind,:) = [Recall_all, Precision_all, AFA,AMT, TP_all,FP_all,FN_all];
    clear bboxes Correct_Det counter det_boxes frame i TP recall FP FN Det_Num_BB gt_boxes recall_mat FN_all GT_all precision precision_mat FP_all groundTruthBboxes TP_all GT_Num_BB
  
    clear AFA AMT Precision_all Recall_all
    
    
end




