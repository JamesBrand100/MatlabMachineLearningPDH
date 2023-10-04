%Load neural network 
net = googlenet

%Look at appropriate input size 
inputSize = net.Layers(1).InputSize

%examine the types of outputs 
%so look at the last layer class names 
classNames = net.Layers(end).ClassNames;
%get # of items, binary output 
numClasses = numel(classNames);
%get 10 random classes for output 
disp(classNames(randperm(numClasses,10)))

%import a picture that i want 
I = imread('bagels.png');
figure
imshow(I)

%look at number of inputs 
size(I)

%resize the data to be involved with 
I = imresize(I,inputSize(1:2));
figure
imshow(I)

%use the network
[label,scores] = classify(net,I);
label

%sort scores and get associated changed indices 
[~,idx] = sort(scores,'descend');
%results from index 5 to index 1, indexing by -1 each time
idx = idx(5:-1:1);
%get associated class names with idx 
classNamesTop = net.Layers(end).ClassNames(idx);
scoresTop = scores(idx);

%get top results 
figure
barh(scoresTop)
xlim([0 1])
title('Top 5 Predictions')
xlabel('Probability')
yticklabels(classNamesTop)

