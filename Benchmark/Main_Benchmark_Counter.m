
% Counting
benchmarkGtDir = 'C:/Users/Kilian/Desktop/Masterarbeit/Code/0_Baseline/people-counting-opencv/videos/';
benchmarkGt_name = "AVG-TownCentre900";

[metsBenchmark] = evaluateCounting('Benchmark_SORT_Parameter.txt', 'res/AVG-TownCentre900/Benchmark_SORT_Parameter/Counter/', benchmarkGtDir,benchmarkGt_name, benchmarkGt_name,"counter_gt.txt");


%C:\Users\Kilian\Desktop\Masterarbeit\Code\0_Baseline\people-counting-opencv\Benchmark\res\AVG-TownCentre900\Benchmark_CORAL_Threshold