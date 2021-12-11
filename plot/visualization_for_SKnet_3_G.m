% visualization for SKnet
% 1. put this file and expRecord.csv in the same dictrionary
% 2. run this code

clear all
% 
% % readCSV
% table = readtable('baseline_5302e-6.csv');
% 
% % extract data
% data = table2array(table(2:5, :));
% iteration = table2array(table(1, :));
% 
% trainLoss = data(1,:);
% trainAcc = data(2,:);
% testLoss = data(3,:);
% testAcc = data(4,:);
[i1 trainl1 traina1 testl1 testa1] = readCSV('sknet5302e-6.csv');
[i2 trainl2 traina2 testl2 testa2] = readCSV('group16.csv');
[i3 trainl3 traina3 testl3 testa3] = readCSV('group64.csv');



figure(1)
plot(i1, trainl1, i1, testl1, i1, trainl2, i1, testl2, i1, trainl3, i1, testl3)
grid minor
title('loss')
legend('G32 train(default)','G32 test(default)','G16 train','G16 test','G64 train','G64 test','Location','southwest')
xlabel('epoch')
ylabel('loss')

figure(2)
plot(i1, testa1, i1, testa2, i1, testa3, i1, 75 * ones(1, 30))
grid minor
title('test accuracy comparison')
legend('G32(default)','G16','G64','75% line', 'Location','southeast')
xlabel('epoch')
ylabel('accuracy')
xlim([0, 30])
ylim([20, 100])

function [iteration, trainL, trainA, testL, testA] = readCSV(filename)
    % readCSV
    table = readtable(filename);

    % extract data
    data = table2array(table(2:5, :));
    iteration = table2array(table(1, :));

    trainL = data(1,:);
    trainA = data(2,:);
    testL = data(3,:);
    testA = data(4,:);
end

