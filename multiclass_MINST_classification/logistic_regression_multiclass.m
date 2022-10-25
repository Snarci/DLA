%% load files
load('.\datasets\MNIST_dataset\MNIST_train_labels.mat')
load('.\datasets\MNIST_dataset\MNIST_train_images.mat')
load('.\datasets\MNIST_dataset\MNIST_test_labels.mat')
load('.\datasets\MNIST_dataset\MNIST_test_images.mat')

%% rename
y=MNIST_train_labels;
X= MNIST_train_images;

y_t=MNIST_test_labels;
X_t=MNIST_test_images;

%% preprocess
[r,c,ch]=size(X);
X=reshape(X,r*c,ch)';
[r,c,ch]=size(X_t);
X_t=reshape(X_t,r*c,ch)';
%% preprocess

X = double(X)/255;
X_t = double(X_t)/255;

%% iter during epochs da fare bene non funziona
theta = zeros(size(X,2),10);
lr = 1e-3;
epochs = 100;
losses= zeros(epochs,1);
losses_test= zeros(epochs,1);
for i=1:epochs
    h_x = X*theta;
    grad = (1/size(X,1)).*(X'*((h_x-y)));
    loss = cost_cross_entropy(X,y,theta);
    losses(i) = loss;
    loss_test=cost_cross_entropy(X_t,y_t,theta);
    losses_test(i) = loss_test;
    theta = theta - lr*grad;
    plot(losses);
end
plot(losses_test);
%% predict old and new data
prediction_train = X*theta;
prediction_train = round(max(prediction_train,[],2));

prediction_test = X_t*theta;
prediction_test = round(max(prediction_test,[],2));



%% print confusion matrix
cm_test = confusionmat(y_t,prediction_test);

cm_train = confusionmat(y,prediction_train);

%% plot losses
plot(losses_test);