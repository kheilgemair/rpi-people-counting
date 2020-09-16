

% % ID Matching
%benchmarkGtDir = 'C:/Users/Kilian/Desktop/Masterarbeit/Benchmark/Mot-Challange Dev Kit/Training Set Data/MOT16Labels/train/';
% [allMets, metsBenchmark] = evaluateTracking('c5-train.txt', 'res/MOT16/data/', benchmarkGtDir, 'MOT16');

% ID Matching
benchmarkGtDir = 'C:/Users/Kilian/Desktop/Masterarbeit/Code/0_Baseline/people-counting-opencv/videos/';
benchmarkGt_name = "AVG-TownCentre";

%name_sequmap = "c5-train";
[allMets, metsBenchmark] = evaluateTracking('AVG-TownCentre/Benchmark_Tracker_Skipped_Frames_Baseline.txt', 'res/AVG-TownCentre/Benchmark_Tracker_Skipped_Frames_Baseline/', benchmarkGtDir,benchmarkGt_name, 'AVG-TownCentre');

allMetsExcel = []
for i = 1:size(allMets,2)
    allMetsExcel(i,:) = allMets(1,i).m;
end
% IniFile = 'C:\Users\Kilian\Desktop\Masterarbeit\Code\0_Baseline\people-counting-opencv\seqinfo_1.ini';
% I = INI('File',IniFile);
% 
% 
% CounterGT = importfile_GT_Counter("C:\Users\Kilian\Desktop\Masterarbeit\Code\0_Baseline\people-counting-opencv\output\AVG-TownCentre\Counter_GT.txt", [1, Inf]);
% CounterGT = table2array(CounterGT);
% Counter_ = importfile_GT_Counter("C:\Users\Kilian\Desktop\Masterarbeit\Code\0_Baseline\people-counting-opencv\output\AVG-TownCentre\Counter.txt", [1, Inf]);
% Counter_ = table2array(Counter_);
% 
% [Recall_Count,Precision_Count] = Count_benchmark(CounterGT,Counter_)