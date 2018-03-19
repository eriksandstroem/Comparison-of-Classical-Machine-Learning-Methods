%Gaussian Classifier with principal component analysis (PCA) - Performs classification on digits
%from a subset containing 5000 training images and 500 test images of the
%MNIST dataset. Classification is done using
%5, 10, 20, 30, 40, 60, 90, 130, 180 and 250 principal
%components in order to compare the performance.
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
%    10 first principal components of digit 5
%    Digit that looks least like a 5
%    10 first principal components over all classes
%    Eigenvalues as a function of PCA number (Scree plot)
%    Error rate over all classes for every choice of #PC used
%    Error rate conditioned on each class for every choice of #PC used
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

% Scaling to reduce risk of overflow
imageTest = imageTest/255; 
imageTrain = imageTrain/255;

%% Find the principal components of digit 5
nbrOfDigit5 = sum((labelTrain == 5));
SampleMean5 = sum(imageTrain(:,:,labelTrain == 5),3)/nbrOfDigit5;

%Find all observations of digit 5 and subtract the mean
X = reshape(imageTrain(:,:,labelTrain == 5)-SampleMean5,[784 nbrOfDigit5]);

%Find the eigenvectors to the covariance matrix and the eigenvalues using
%the Singular Value Decomposition (SVD)
[U,S,V] = svd(X','econ');

%Plot the 10 first principal components
figure
for i = 1:10
   subplot(5,2,i);
   imshow(reshape(V(:,i),[28 28]),[]);
   title(['Component ' num2str(i)]); 
end

% Display the digit that is looks least like a 5. 40 PC is assumed.
imageTestTemp = imageTest(:,:,1);
imageTestTempVec = imageTestTemp(:)-SampleMean5(:);
x = V'*imageTestTempVec;
y = x(1:40);
normsquaredY = y'*y;
normsquaredZ = x'*x-normsquaredY; 
index = 1;
for i = 2:500
    imageTestTemp = imageTest(:,:,i);
    imageTestTempVec = imageTestTemp(:)-SampleMean5(:);
    x = V'*imageTestTempVec;
    y = x(1:40);
    normsquaredY = y'*y;
    normsquaredZTemp = x'*x-normsquaredY; 
        if normsquaredZTemp > normsquaredZ
            index = i;
            normsquaredZ = normsquaredZTemp;
        end
end 

figure
imshow(imageTest(:,:,index),[]);
title(['Index: ' num2str(index)]);
%% Find the principal components and eigenvalues over all classes combined
SampleMean = sum(imageTrain(:,:,:),3)/5000;

%Find all observations of digit 5 and subtract the mean
X = reshape(imageTrain(:,:,:)-SampleMean,[784 5000]);

%Find the eigenvectors to the covariance matrix and the eigenvalues using
%the Singular Value Decomposition (SVD)
[U,S,V] = svd(X','econ');

%Plot the 10 first principal components
figure
for i = 1:10
   subplot(5,2,i);
   imshow(reshape(V(:,i),[28 28]),[]);
   title(['Component ' num2str(i)]); 
end

%Plot the eigenvalues
figure
plot(1:784,sum(S));
title('Eigenvalues as a function of principal component over all classes');
ylabel('Eigenvalue');
xlabel('Principal component number');

%% Transform all images to the PC subspace and process in the subspace
Y = V'*X;
%Save only the most important PC, given by k
k = [5, 10, 20, 30, 40, 60, 90, 130, 180, 250];
digitsPerClassTrain = zeros(1,10); %Digits per class in training set
digitsPerClassTest = zeros(1,10); %Digits per class in test set
for i = 1:10
    digitsPerClassTrain(i) = sum(labelTrain == i-1);
    digitsPerClassTest(i) = sum(labelTest == i-1);
end
P_YC = digitsPerClassTrain/5000;
nbrOfErrorsPerClass = zeros(length(k),10);
errorPerClass = zeros(length(k),10);
classifier = zeros(500,length(k));
totalError = zeros(length(k),1); %Error rate over all classes for each choice of #PC


%% Perform Gaussian Classification on different number of PC, given by k
for q = 1:length(k)
Y_hat = Y(1:k(q),:); %Truncated version of Y where only k(q) PC are used
% Calculate the sample mean and covariance for each digit class
SampleMeanSubSpace = zeros(k(q),10);
CovMtx = zeros(k(q),k(q),10);
for i = 1:10
    SampleMeanSubSpace(:,i) = sum(Y_hat(:,labelTrain == i-1),2)/digitsPerClassTrain(i);
    CovMtx(:,:,i) = cov(Y_hat(:,(labelTrain == i-1))');
end

% Gaussian classification
for p = 1:500
    imageTestTemp = imageTest(:,:,p);
    imageTestTemp = V'*(imageTestTemp(:)-SampleMean(:));
    imageTestTempVec = imageTestTemp(1:k(q));
    Gaussian_Eval = P_YC(1)*mvnpdf(imageTestTempVec, SampleMeanSubSpace(:,1),CovMtx(:,:,1));
    classifier(p,q) = 0;
    for j = 2:10
        Gaussian_Eval_Temp = P_YC(j)*mvnpdf(imageTestTempVec, SampleMeanSubSpace(:,j),CovMtx(:,:,j));
        if Gaussian_Eval_Temp > Gaussian_Eval
            Gaussian_Eval = Gaussian_Eval_Temp;
            classifier(p,q) = j-1;
        end
    end 
end

% Determine total error rate over all classes
totalError(q) = sum((labelTest-classifier(:,q) ~= 0))/500;

% Determine number of digits within each class in the test set and number of errors per class
for i = 1:10
    classifierTemp = classifier(:,q);
    nbrOfErrorsPerClass(q,i) = sum(classifierTemp(labelTest == i-1) ~= i-1);
    errorPerClass(q,i) = nbrOfErrorsPerClass(q,i)/digitsPerClassTest(i);
end

%Plot conditional probability error rate per class
subplot(2,5,q);
bar(0:9,errorPerClass(q,:));
title(['#PC ' num2str(k(q))]);
xlabel('Class number');
ylabel('P(Error|Class = i)');

end

figure
bar(k,totalError);
title('Total error as a function of the number of principal components');
xlabel('Number of principal components');
ylabel('Total error over all classes');

%------------- END OF CODE --------------