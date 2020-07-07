% For more information or enquiries
% Contact author at ahmedakano@gmail.com. 08106621695.


% Matlab Program to Train a neural network To predict Baby's Weight
% using Levenberg-Marquardt Back Propagation Algorithm 
% improved with Genetic Algorithm and Simulated Annealing.

data = xlsread('babyData.xls'); % read the excel file "babyData.xls"

%mother_height = data(:,1);
%mother_weight = data(:,2);
%mother_age = data(:,3);

%baby_weight = data(:,4);

%inputs = [mother_height; mother_weight; mother_age] ;


inputs = data(:,1:3); % extract input columns
targets = data(:,4); % extract target column


inputs = inputs'; % transpose input matrix
targets = targets';

%{	 
code to determine the best hiddenLayerSize for the network
time-consuming operation 
for i=1:100  vary number of hidden layer neurons from 1 to 100
    
	hiddenLayerSize = i; %number of hidden layer neurons
    net = feedforwardnet(hiddenLayerSize);

	net.divideParam.trainRatio = .7;
	net.divideParam.valRatio = .15;
	net.divideParam.testRatio = .15;

	[net, tr] = train(net, inputs, targets) ;

	outputs = net(inputs(tr.testInd));

	rmse(i) = sqrt(mean((outputs-targets(tr.testInd)).^2)) ;
end

[minRmse, minIndex] = min(rmse);
hiddenLayerSize = minIndex ;
%}

hiddenLayerSize = 15 ; % 15 is randomly hard-coded as hiddenLayerSize.

trainFcn = 'trainlm' ; % using Levenberg-Marquardt training algorithm
%trainFcn = 'traingd' ; % uncomment to use gradient descent training algorithm

net = feedforwardnet(hiddenLayerSize, trainFcn) ; % create ff network

net.divideParam.trainRatio = .7; % 70 per cent of data for training
net.divideParam.valRatio = .15; % 15 per cent for validation
net.divideParam.testRatio = .15; % 15 per cent for testing

[net, tr] = train(net, inputs, targets) ; % train network with data

outputs = net(inputs(tr.testInd)); % use trained network on test data
rmseBP = sqrt(mean((outputs-targets(tr.testInd)).^2)); % calculate error

%{
The remaining part of the code is for improving the neural network
with generatic algorithm and simulated annealing algorithms.
%}

weightsNbias = getwb(net); % get weightsNbias from trained network
weightsNbias = weightsNbias' ; % transpose vector

h = @(x) mse_test(x, net, inputs, targets);

% stopping criterion can be improved
% ga should be allowed more generations for better result
ga_opts = gaoptimset('Generations',20, 'display','iter', 'InitialPopulation', weightsNbias);

[x_ga_opt, err_ga] = ga(h, 5*hiddenLayerSize+1, ga_opts);

net = setwb(net, x_ga_opt'); % set network to use GA improved weightsNbias.

outputs = net(inputs(tr.testInd)); % use GA improved network on test data
rmseGA = sqrt(mean((outputs-targets(tr.testInd)).^2)) ; % calculate error

% stopping criterion can be improved
% 'MaxIter' can be increased.
sa_opts = saoptimset('MaxIter',40, 'display','iter');

initialWB = x_ga_opt ;
[x_sa_opt, fval] = simulannealbnd(h, initialWB, [], [], sa_opts) ;

net = setwb(net, x_sa_opt') ; % set network to use SA improved weightsNbias.

outputs = net(inputs(tr.testInd)); % use SA improved network on test data
rmseSA = sqrt(mean((outputs-targets(tr.testInd)).^2)) ; % calculate error

fprintf('\nRoot mean square error of trained network using Levenberg-Marquardt : %f \n',rmseBP);
fprintf('\nRoot mean square error after network is improved using Genetic Algorithm : %f\n',rmseGA);
fprintf('\nRoot mean square error after network is further improved with Simulated Annealing : %f\n',rmseSA);


% save(['trainedNetwork'], 'net') % uncomment to saved the trained network named 'net' as 'trainedNetwork.mat';