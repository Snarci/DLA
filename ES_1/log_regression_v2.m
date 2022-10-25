%% load files
load('C:\Users\39342\Desktop\DLA\ES_1\MNIST_dataset\MNIST_train_labels.mat')
load('C:\Users\39342\Desktop\DLA\ES_1\MNIST_dataset\MNIST_train_images.mat')
load('C:\Users\39342\Desktop\DLA\ES_1\MNIST_dataset\MNIST_test_labels.mat')
load('C:\Users\39342\Desktop\DLA\ES_1\MNIST_dataset\MNIST_test_images.mat')

%% filter ones and zeroes
y=MNIST_train_labels(MNIST_train_labels == 0 | MNIST_train_labels == 1 );
X= MNIST_train_images(:,:,MNIST_train_labels == 0 | MNIST_train_labels == 1 );

y_t=MNIST_test_labels(MNIST_test_labels == 0 | MNIST_test_labels == 1 );
X_t=MNIST_test_images(:,:,MNIST_test_labels == 0 | MNIST_test_labels == 1 );

%% preprocess
[r,c,ch]=size(X);
X=reshape(X,r*c,ch)';
[r,c,ch]=size(X_t);
X_t=reshape(X_t,r*c,ch)';
%% preprocess

X = double(X)/255;
X_t = double(X_t)/255;

%% iter during epochs
theta = zeros(size(X,2),1);
lr = 1e-3;
epochs = 10;
losses= zeros(epochs,1);
losses_test= zeros(epochs,1);
for i=1:epochs
    h_x = sigmoid(X*theta);
    grad = (1/size(X,2))*(X'*((h_x-y)));
    loss = cost_function(X,y,theta);
    losses(i) = loss;
    loss_test=cost_function(X_t,y_t,theta);
    losses_test(i) = loss_test;
    theta = theta - lr*grad;
    plot(losses);
end

%% predict old and new data
prediction_train = sigmoid(X*theta);
prediction_train(prediction_train >= 0.5 ) = 1;
prediction_train(prediction_train ~= 1) = 0;

prediction_test = sigmoid(X_t*theta);
prediction_test(prediction_test >= 0.5 ) = 1;
prediction_test(prediction_test ~= 1) = 0;

%% print confusion matrix
cm_test = confusionmat(y_t,prediction_test);
tp = cm_test(1,1);
fp = cm_test(1,2);
fn = cm_test(2,1);
tn = cm_test(2,2);
accuracy_test=(tp+tn)/(tp+tn+fp+fn);

cm_train = confusionmat(y,prediction_train);
tp = cm_train(1,1);
fp = cm_train(1,2);
fn = cm_train(2,1);
tn = cm_train(2,2);
accuracy_train=(tp+tn)/(tp+tn+fp+fn);
%% plot losses
plot(losses_test);