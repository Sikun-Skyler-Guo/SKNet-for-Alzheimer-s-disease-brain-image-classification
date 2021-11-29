% visualization for SKnet
% 1. put this file and expRecord.csv in the same dictrionary
% 2. run this code

clear all

% readCSV
table = readtable('expRecord.csv');

% extract data
data = table2array(table(2:5, :));
iteration = table2array(table(1, :));

trainLoss = data(1,:);
trainAcc = data(2,:);
testLoss = data(3,:);
testAcc = data(4,:);

figure(1)
plot(iteration, trainLoss, iteration, testLoss)
title('loss vs epoch')
legend('train loss','test loss','Location','southeast')
xlabel('epoch')
ylabel('loss')

figure(2)
plot(iteration, trainAcc, iteration, testAcc)
title('accuracy vs epoch')
legend('train accurcy', 'test accuracy','Location','southeast')
xlabel('epoch')
ylabel('accuracy')
