% 
% url_det = 'C:\Users\Kilian\Desktop\Masterarbeit\Code\0_Baseline\people-counting-opencv\Benchmark\res\AVG-TownCentre\Benchmark_Detection\Det\AVG-TownCentre_Detection_1.txt';
% H_img = 1080;
% W_img = 1920;

% %Ground Truth Heat map
% url_det = 'C:\Users\Kilian\Desktop\Masterarbeit\Code\0_Baseline\people-counting-opencv\videos\AVG-TownCentre\gt\gt.txt';
% H_img = 1080;
% W_img = 1920;

% url_det = 'C:\Users\Kilian\Desktop\Masterarbeit\Code\0_Baseline\people-counting-opencv\Benchmark\res\MOT16-04\Benchmark_Detection\Det\MOT16-04.txt';
% H_img = 540;
% W_img = 960;

%Ground Truth Heat map
url_det = 'C:\Users\Kilian\Desktop\Masterarbeit\Code\0_Baseline\people-counting-opencv\videos\MOT16-04\gt\gt.txt';
H_img = 540;
W_img = 960;


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


det =  readtable(url_det, opts);

det = table2array(det);



%det = Sort;

clear url_gt url_det url_Sort opts


count_detections = zeros(H_img,W_img);
proba_detections = zeros(H_img,W_img);
results = zeros(H_img,W_img);

for counter = 1:size(det,1)     

    det_boxes = find(det(:,1)==counter);

    bboxes = det(det_boxes,3:7);


    
    for i = 1:size(bboxes,1)
        
        confidence = bboxes(i,5);

        x = int16(round(bboxes(i,1)) +1);
        if x <= 0
            disp("--------Counter------")
            disp(counter)
            disp("X-Problem Negativ")
            disp(x);  
            disp("----------------------")
            x = 1;
        end
        
        y = int16(round(bboxes(i,2)) +1);
        if y <= 0
            disp("--------Counter------")
            disp(counter)
            disp("Y-Problem Negativ")
            disp(y);  
            disp("----------------------")
            y = 1;
        end
        
        w = int16(round(bboxes(i,3)));
        h = int16(round(bboxes(i,4)));
        
        endX = w + x;
        
        if endX > W_img
            disp("--------Counter------")
            disp(counter)
            disp("X-Problem zu groﬂ")
            disp(endX);  
            disp("----------------------")
            endX = W_img;
        end
        
        endY = h + y;
    
        if endY > H_img
            disp("--------Counter------")
            disp(counter)
            disp("Y-Problem zu groﬂ")
            disp(endY);  
            disp("----------------------")
            disp(endY);   
            endY = H_img;
        end
        
        count_detections(y:(endY),x:(endX)) = count_detections(y:(endY),x:(endX)) + 1;
        proba_detections(y:(endY),x:(endX)) = proba_detections(y:(endY),x:(endX)) + confidence;
    end
end 


for i = 1:size(results,1)  
    for j = 1:size(results,2)  
        
        results(i,j) = proba_detections(i,j) / count_detections(i,j);
        
    end
end


imagesc(count_detections)
colorbar


Max_line = max(count_detections);



max(Max_line)
