function [Recall,Precision] = Count_benchmark(CounterGT,Counter_)


sum_up_frames = 60;

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
    
    Sum_up_frame_Counter = Counter_(i,2) + Sum_up_frame_Counter;
    Sum_down_frame_Counter = Counter_(i,3) + Sum_down_frame_Counter;
    
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
        
        if (Sum_down_frame_GT - Sum_down_frame_Counter) > 0
            FN_down = (Sum_down_frame_GT - Sum_down_frame_Counter) + FN_down;
            TP_down = Sum_down_frame_Counter + TP_down;
        elseif (Sum_up_frame_GT - Sum_up_frame_Counter) == 0
            TP_down = Sum_down_frame_GT + TP_down;
            
        elseif (Sum_down_frame_GT - Sum_down_frame_Counter) < 0
            FP_down = abs(Sum_down_frame_GT - Sum_down_frame_Counter) + FP_down;
            TP_down = Sum_down_frame_GT + TP_down;
        end
                
        Sum_up_frame_GT = 0;
        Sum_down_frame_GT = 0;
    
        Sum_up_frame_Counter = 0;
        Sum_down_frame_Counter = 0;
        
    end
end


TP = TP_up + TP_down;
FP = FP_up + FP_down;
FN = FN_up + FP_down;

counted = sum(Counter_(:,2)) + sum(Counter_(:,3));
GT = sum(CounterGT(:,2)) + sum(CounterGT(:,3));

Recall = TP / (TP + FN);

Precision = TP / (TP + FP);

TP_GT = counted / GT;



end

