%Gaussian Classifier - Performs classification on digits
%from a subset containing 5000 training images and 500 test images of the
%MNIST dataset. It is assumed that the random variables constructing the
%images are independent such that the covariance is the identity
%and also that the prior for the classes are distributed uniformly, i.e
%PY(i) = 1/N = constant where N is the number of classes. Please take a
%look at the Gaussian classifier with PCA in order to make use of the
%covariance.
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
%    classifier - vector containing results of NN classification
%    covMtx - 10 covariance matrices, one for each class
%    totalError - error rate over all classes
%    errorPerClass - error rate per class
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

% Calculate the sample mean and covariance for each image class
SampleMean = zeros(28,28,10);
digitsPerClassTrain = zeros(1,10); %Digits per class in training set
CovMtx = zeros(784,784,10);
for i = 1:10
    digitsPerClassTrain(i) = sum((labelTrain == i-1));
    SampleMean(:,:,i) = sum(imageTrain(:,:,labelTrain == i-1),3)/digitsPerClassTrain(i);
    X = reshape(imageTrain(:,:,labelTrain == i-1),[784 digitsPerClassTrain(i)]);
    CovMtx(:,:,i) = cov(X');
end

% Plot sample mean for each class
for i = 1:10
subplot(2,5,i);
imshow(SampleMean(:,:,i), []); 
title(['Class ' num2str(i-1) ' mean']);
end

% Plot covariance matrix for each class
figure
for i = 1:4
subplot(2,2,i);
imshow(CovMtx(:,:,i), []); 
title(['Covariance matric of class ' num2str(i-1)]); 
end
figure
for i = 5:8
subplot(2,2,i-4);
imshow(CovMtx(:,:,i), []); 
title(['Covariance matric of class ' num2str(i-1)]); 
end
figure
for i = 9:10
subplot(1,2,i-8);
imshow(CovMtx(:,:,i), []); 
title(['Covariance matric of class ' num2str(i-1)]); 
end

%% Gaussian Classifier
classifier = zeros(500,1); % Classification vector
for k = 1:500
    imageTestTemp = imageTest(:,:,k);
    imageTestTempVec = imageTestTemp(:);
    SampleMeanMatrix = SampleMean(:,:,1);
    SampleMeanVec = SampleMeanMatrix(:);
    Gaussian_Eval =  -1/2*((imageTestTempVec-SampleMeanVec)'*(imageTestTempVec-SampleMeanVec));
    classifier(k) = 0;
    for j = 2:10
        SampleMeanMatrix = SampleMean(:,:,j);
        SampleMeanVec = SampleMeanMatrix(:);
        Gaussian_Eval_Temp =  -1/2*((imageTestTempVec-SampleMeanVec)'*(imageTestTempVec-SampleMeanVec));
        if Gaussian_Eval_Temp > Gaussian_Eval
            Gaussian_Eval = Gaussian_Eval_Temp;
            classifier(k) = j-1;
        end
    end 
end

%% Determine total error rate over all classes
totalError = sum((labelTest-classifier ~= 0))/500;

%% Determine number of digits within each class in the test set and number of errors per class
digitsPerClass = zeros(1,10);
nbrOfErrorsPerClass = zeros(1,10);
errorIndex = find(labelTest-classifier ~= 0); %Indices in labelTest that are classified falsely
errorPerClass = zeros(1,10);
for i = 1:10
    digitsPerClass(i) = sum((labelTest == i-1));
    nbrOfErrorsPerClass(i) = sum(classifier(labelTest == i-1) ~= i-1);
    errorPerClass(i) = nbrOfErrorsPerClass(i)/digitsPerClass(i);
end

%Plot conditional probability error rate per class
figure
bar(0:9,errorPerClass);
title('Plot of the conditional probability error rates as a function of the classes');
xlabel('Class number');
ylabel('P(Error|Class = i)');

%------------- END OF CODE --------------

