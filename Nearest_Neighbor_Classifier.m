%Nearest Neighbor (NN) Algorithm - Performs NN classification on digits
%from a subset containing 5000 training images and 500 test images of the
%MNIST dataset.
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
%    closestNeighborMatrix - matrix containing information of the closest
%    neighbor for every test image in the training set
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

clear all
close all
load data.mat
load label.mat
%% Define vectors and matrices for storage
classifier = zeros(500,1);
closestNeighborMatrix = zeros(500,2);
closestNeighborMatrix(:,1) = 1:500;

%% Nearest Neighbor Algorithm
for i = 1:500
    d = sqrt(sum(sum(abs(imageTrain(:,:,1)-imageTest(:,:,i)).^2)));
    classifier(i) = labelTrain(1);
   for k = 2:5000 
       dtemp = sqrt(sum(sum(abs(imageTrain(:,:,k)-imageTest(:,:,i)).^2)));
       if(dtemp < d)
          d = dtemp;
          classifier(i) = labelTrain(k);
          closestNeighborMatrix(i,2) = k;
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
title('Nearest Neighbor');
xlabel('Class number');
ylabel('P(Error|Class = i)');
ylim([0 0.35])

%Plot five examples of misclassified digits
for i = 1:5
figure
subplot(2,1,1)
imshow(imageTest(:,:,errorIndex(9*i)),[]);
title(['Misclassified image, imageTest index = ' num2str(errorIndex(9*i))]);
hold on
subplot(2,1,2);
imshow(imageTrain(:,:,closestNeighborMatrix(errorIndex(9*i),2)),[]);
title(['Training image, imageTrain index = ' ...
    num2str(closestNeighborMatrix(errorIndex(9*i),2))]);
end

%------------- END OF CODE --------------

