

gt_name = "gt.txt";

% Counting
benchmarkGtDir = 'C:/Users/Kilian/Desktop/Masterarbeit/Code/0_Baseline/people-counting-opencv/videos/';
benchmarkGt_name = "AVG-TownCentre900";

[metsBenchmark] = evaluateDetection_1('AVG-TownCentre900/Benchmark_SORT_Parameter.txt', 'res/AVG-TownCentre900/Benchmark_SORT_Parameter/Det/', benchmarkGtDir,benchmarkGt_name, benchmarkGt_name,gt_name);

