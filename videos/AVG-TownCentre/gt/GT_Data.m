
% 
% The ground truth and tracking output is provided in the '.top' file format. This consists of rows in comma-seperated variable (CSV) format:
% 
% personNumber, frameNumber, headValid, bodyValid, headLeft, headTop, headRight, headBottom, bodyLeft, bodyTop, bodyRight, bodyBottom
% 
% personNumber - A unique identifier for the individual person
% frameNumber - The frame number (counted from 0)
% headValid - 1 if the head region is valid, 0 otherwise
% bodyValid - 1 if the body region is valid, 0 otherwise
% headLeft,headTop,headRight,headBottom - The head bounding box in pixels
% bodyLeft,bodyTop,bodyRight,bodyBottom - The body bounding box in pixels

%
% 
% 1 Frame number Indicate at which frame the object is present 
% 2 Identity number Each pedestrian trajectory is identi?ed by a unique ID (?1 for detections) 
% 3 Bounding box left Coordinate of the top-left corner of the pedestrian bounding box 
% 4 Bounding box top Coordinate of the top-left corner of the pedestrian bounding box 
% 5 Bounding box width Width in pixels of the pedestrian bounding box 
% 6 Bounding box height Height in pixels of the pedestrian bounding box 
% 7 Con?dence score DET: Indicates how con?dent the detector is that this instance is a pedestrian. 


fileID = fopen('gt.txt','w');

v = VideoReader('TownCentreXVID.avi');

town = TownCentregroundtruth;

gt(:,1) = int16(town(:,2) +1);    %GGf. Plus 1 (FRAME 0)
gt(:,2) = int16(town(:,1) +1);    % ID

gt(:,3) = town(:,9);
gt(:,4) = town(:,10);

gt(:,5) = town(:,11)-town(:,9);
gt(:,6) = town(:,12)-town(:,10);


counter = 0;

while hasFrame(v)
        
    counter = counter + 1;  
    
    gt_boxes = find(gt(:,1)==counter);
    %det_boxes = find(det(:,1)==counter);
    
    groundTruthBboxes = gt(gt_boxes,3:6);
    %bboxes = det(det_boxes,3:6);
      
    groundTruthBboxes_ID = gt(gt_boxes,2);
    %[precision,recall] = bboxPrecisionRecall(bboxes,groundTruthBboxes,0.5);
    
    
    frame = readFrame(v);
    %if counter > 4000
    
    %image(frame)
    %hold on
   
    % Red Bounding Boxes
    for i=1:size(groundTruthBboxes,1)
        fprintf(fileID,'%d,%d,%.2f,%.2f,%.2f,%.2f,1,1,1\n',counter,groundTruthBboxes_ID(i,1),groundTruthBboxes(i,1),groundTruthBboxes(i,2),groundTruthBboxes(i,3),groundTruthBboxes(i,4));
        %rectangle('Position',groundTruthBboxes(i,:),'EdgeColor','r');
    end
    %pause(0.05)
    %end
     
    if counter == 4501
        break     
    end

    
end

fclose(fileID);
