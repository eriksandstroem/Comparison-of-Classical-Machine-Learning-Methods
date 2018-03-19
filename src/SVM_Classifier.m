%Support Vector Machine (SVM) - Performs classification on digits
%from a subset containing 5000 training images and 500 test images of the
%MNIST dataset. The LIBSVM library is used. First, two class classification
%is tested on digits 6 and 8 with a linear kernel and later multiclass classification is used
%with the radial basis function as kernel. Make sure to download and install the
%library before executing the file. Put all mex-files in the current
%directory.
%
% Need to be loaded:
%
%    imageTest - Download here: http://www.svcl.ucsd.edu/courses/ece175/
%    imageTrain - Download here: http://www.svcl.ucsd.edu/courses/ece175/
%    labelTest - Download here: http://www.svcl.ucsd.edu/courses/ece175/
%    labelTrain - Download here: http://www.svcl.ucsd.edu/courses/ece175/
%
% Outputs:
%    
%    Binary classification analysis with linear kernel of digits 6 and 8 including
%       - Normal vector to the decision boundary as a 28x28 image
%       - Five support vectors from each class as 28x28 images
%       - The five digits in the test set from each class closest to the
%       decision boundary
%       - The five digits in the test set from each class farthest from the
%       decision boundary
%   
%    Multiclass classification with RBA kernel on all digits including
%       - 2-fold cross validation and grid search to find optimal parameter values
%       - Classification results
%    
% Other m-files required: none
% Subfunctions: none
%
% Author: Erik Sandström, Lund University
% email address: erik.sandstrm@gmail.com
% Website: http://www.svcl.ucsd.edu/courses/ece175/
% March 2018

%------------- BEGIN CODE --------------
close all
clear all
load data.mat
load label.mat

% Create training and test sets only containing digits 6 and 8
trainingSet68 = imageTrain(:,:,labelTrain == 6 | labelTrain == 8);
labelTrainSet68 = labelTrain(labelTrain == 6 | labelTrain == 8);
testSet68 = imageTest(:,:,labelTest == 6 | labelTest == 8);
labelTestSet68 = labelTest(labelTest == 6 | labelTest == 8);

trainingSet68 = reshape(trainingSet68,[784 size(trainingSet68,3)])';
testSet68 = reshape(testSet68,[784 size(testSet68,3)])';

% SVM options
svmopts=('-t 0 -c 2^(-4)');
% train SVM on training data
model = svmtrain(labelTrainSet68,trainingSet68, svmopts);

% test SVM on training data
[trainingSet68out, trainingSet68Acc, trainingSet68ext]=svmpredict(labelTrainSet68, trainingSet68, model);

% test SVM on test data
[testSet68out, testSet68Acc, testSet68ext]=svmpredict(labelTestSet68, testSet68, model);

% Calculate and display the normal vector omega to the decision boundary
omega = model.SVs'*model.sv_coef;
figure
imshow(reshape(omega,[28,28]),[]);
title('Normal vector \omega to the decision boundary');

% Divide the support vectors to the two classes
SVs6 = trainingSet68(model.sv_indices(labelTrainSet68(model.sv_indices) == 6),:);
SVs8 = trainingSet68(model.sv_indices(labelTrainSet68(model.sv_indices) == 8),:);

figure
% Digits in training set closest to decision boundary
for i = 1:5
subplot(2,5,i);
imshow(reshape(SVs6(i,:),[28 28]),[]);
title(['Nbr ' num2str(i)]);
subplot(2,5,i+5);
imshow(reshape(SVs8(i,:),[28 28]),[]);
title(['Nbrt ' num2str(i)]);
end

% Display they histogram of the distance to the decision boundary for each
% test image
b = -model.rho;
histogram(abs(omega'*testSet68'+b*ones(1,83)),10);
title('Histogram of distance to the decision boundary');
ylabel('Number of test samples');
xlabel('Distance from decision boundary');

%% Plot the 5 digits in the test set closest to and farthest from the decision boundary for each class

testSet6 = testSet68(labelTestSet68 == 6,:);
testSet8 = testSet68(labelTestSet68 == 8,:);

keySetDist6 = abs(omega'*testSet6'+b*ones(1,size(testSet6,1)));
keySetDist8 = abs(omega'*testSet8'+b*ones(1,size(testSet8,1)));
valueSet6 = 1:size(testSet6,1);
valueSet8 = 1:size(testSet8,1); 

mapObj6 = containers.Map(keySetDist6,valueSet6);
mapObj8 = containers.Map(keySetDist8,valueSet8);

[sortedKeys6,sortIdx6] = sort(cell2mat(mapObj6.keys),'descend');
[sortedKeys8,sortIdx8] = sort(cell2mat(mapObj8.keys),'descend');

sortedValues6 = cell2mat(values6(sortIdx6));
sortedValues8 = cell2mat(values8(sortIdx8));

figure
% Digits in test set farthest from the decision boundary
for i = 1:5
subplot(2,5,i);
imshow(reshape(testSet6(sortedValues6(i),:),[28 28]),[]);
title(['Nbr ' num2str(i)]);
subplot(2,5,i+5);
imshow(reshape(testSet8(sortedValues8(i),:),[28 28]),[]);
title(['Nbr ' num2str(i)]);
end

figure
% Digits in test set closest to the decision boundary
for i = 1:5
subplot(2,5,i);
imshow(reshape(testSet6(sortedValues6(length(sortedValues6)-i+1),:),[28 28]),[]);
title(['Nbr ' num2str(i)]);
subplot(2,5,i+5);
imshow(reshape(testSet8(sortedValues8(length(sortedValues8)-i+1),:),[28 28]),[]);
title(['Nbr ' num2str(i)]);
end

%% Perform multiclass classification on all digits using the radial basis function kernel

% Perform 2-fold cross validation and grid search to find the optimal
% parameters of C and gamma before classification.
C = [2^(-3) 2^(-1) 2 2^3 2^5 2^7 2^9 2^11];
gamma = [2^(-11) 2^(-9) 2^(-7) 2^(-5) 2^(-3) 2^(-1)];
accMatrix = zeros(length(C),length(gamma)); 
Set1 = reshape(imageTrain(:,:,1:2500),[784, 2500])'/255;
labelSet1 = labelTrain(1:2500);
Set2 = reshape(imageTrain(:,:,2501:5000),[784, 2500])'/255;
labelSet2 = labelTrain(2501:5000);
for i = 1:length(C)
    string1 = num2str(C(i));
    for k = 1:length(gamma)
        string2 = num2str(gamma(k));
        svmopts=(['-t 2 -q -c ' string1 ' -g ' string2]);
        model = svmtrain(labelSet1,Set1, svmopts);
        [tout, tAcc, text]=svmpredict(labelSet2, Set2, model);
        error = tAcc(1);
        model = svmtrain(labelSet2,Set2, svmopts);
        [tout, tAcc, text]=svmpredict(labelSet1, Set1, model);
        accMatrix(i,k) = 1/2*(error+tAcc(1));
    end
end

%% Running the grid search above yields gamma = 2^(-5) and c = 2^11
c=2^11;
g=2^-5;
svmopts = ['-t 2 -c ',num2str(c),' -g ',num2str(g)];
TrainSet = reshape(imageTrain,[784, 5000])'/255;
model = svmtrain(labelTrain,TrainSet, svmopts);
TestSet = reshape(imageTest,[784, 500])'/255;
[tout, tAcc, text]=svmpredict(labelTest, TestSet, model);

%% Determine total error rate over all classes
totalError = sum((labelTest-tout ~= 0))/500;

%% Determine number of digits within each class in the test set and number of errors per class
digitsPerClassTest = zeros(1,10);
nbrOfErrorsPerClass = zeros(1,10);
errorIndex = find(labelTest-tout ~= 0); %Indices in labelTest that are classified falsely
errorPerClass = zeros(1,10);
for i = 1:10
    digitsPerClassTest(i) = sum((labelTest == i-1));
    nbrOfErrorsPerClass(i) = sum(tout(labelTest == i-1) ~= i-1);
    errorPerClass(i) = nbrOfErrorsPerClass(i)/digitsPerClassTest(i);
end

%Plot conditional probability error rate per class
figure
bar(0:9,errorPerClass);
title('SVM');
xlabel('Class number');
ylabel('P(Error|Class = i)');
ylim([0 0.35])
%------------- END OF CODE --------------


