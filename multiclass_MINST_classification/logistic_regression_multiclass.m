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
lr = 1e-1;
epochs = 150;
losses= zeros(epochs,1);
losses_test= zeros(epochs,1);
one_hot_encoding = one_hot_encoding(y,10);
one_hot_encoding_t = one_hot_encoding(y_t,10);
mu= 0.01;
for i=1:epochs
    x_h = X*theta;
    x_h = soft_max(x_h);
    grad = ((1/10)*(X'*(one_hot_encoding-x_h)));
    loss = cost_cross_entropy(x_h,one_hot_encoding);
    losses(i) = loss;
    x_h_t = X_t*theta;
    x_h_t = soft_max(x_h_t);
    loss_test = cost_cross_entropy(x_h_t,one_hot_encoding_t);
    losses_test(i) = loss_test;
    theta = theta + lr*grad;
    plot(losses);
end
plot(losses);
%% predict old and new data
prediction_train = X*theta;
prediction_train = soft_max(prediction_train);
[~,prediction_train] = max(prediction_train,[],2);
prediction_train=prediction_train-1;

prediction_test = X_t*theta;
prediction_test = soft_max(prediction_test);
[~,prediction_test] = max(prediction_test,[],2);
prediction_test=prediction_test-1;



%% print confusion matrix
cm_test = confusionmat(y_t,prediction_test);

cm_train = confusionmat(y,prediction_train);

%% plot losses
plot(losses_test);