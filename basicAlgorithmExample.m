
%load data in 
load fisheriris

%set up figure 
f = figure;

%scatter and label 
gscatter(meas(:,1), meas(:,2), species,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');

%linear discriminator setup 
lda = fitcdiscr(meas(:,1:2),species);

%predict the data that we trained on 
ldaClass = resubPredict(lda);

%get the error on the training set
ldaResubErr = resubLoss(lda)

%create figure showing classification of training data  
figure(f)
bad = ~strcmp(ldaClass,species);
hold on;
plot(meas(bad,1), meas(bad,2), 'kx');
hold off;

%create dummy data points to show decision boundary 
[x,y] = meshgrid(4:.1:8,2:.1:4.5);

%transpose of each 
x = x(:);
y = y(:);

%classify each and scatter the results 
j = classify([x y],meas(:,1:2),species);
gscatter(x,y,j,'grb','sod')