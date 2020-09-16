function [heatmap_detections] = evaluateHeatMap(seqmap, resDir, gtDataDir,benchmarkGt_name, benchmark, gt_name)

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

    
    count_detections = zeros(1080,1920);

    for counter = 1:1500      

        det_boxes = find(det(:,1)==counter);

        bboxes = det(det_boxes,3:6);
        disp(counter)
        for i = 1:size(bboxes,1)
           
            x = round(bboxes(i,1)) +1;
            y = round(bboxes(i,2)) +1;
            w = round(bboxes(i,3));
            h = round(bboxes(i,4));
            
            disp("---------")          
 
            disp("---------")
            if (w+h) > 1920
                disp("X-Problem")
                disp(x);  
                disp(x+w);               
            else
                disp("X")
                disp(x);  
                disp(x+w);  
            end
 
            if (y+h) > 1080
                disp("Y- Problem")
                disp(y+h);              
            else           
                disp("Y")
                disp(y+h);              
            end
            disp(i)
            count_detections(y:(y+h),x:(x+w)) = 1;
            
        end
        
        

    end 
    
    heatmap_detections = 0;
    
end

