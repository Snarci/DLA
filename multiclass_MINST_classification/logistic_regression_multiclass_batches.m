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

%% iter during epochs 
%parametri di controllo
num_classes=10;
lr = 1e-3;
epochs = 1000;
%inizializzazione matrici
losses= zeros(epochs,1);
losses_test= zeros(epochs,1);
theta = zeros(size(X,2),num_classes);
one_hot_encoded = one_hot_encoding(y,num_classes);
one_hot_encoded_t = one_hot_encoding(y_t,num_classes);
%variabili salva best epoch
best_epoch_weigths=theta;
best_train_loss=Inf;
best_test_loss=Inf;

nbatches = 8;
batch_size=floor(size(X,1)/nbatches);
for i=1:epochs
    if (mod(i,100)==0)
        disp(i);
    end
    x_h=[];
    for j=1:nbatches
        if j==nbatches
            range=(1+(j-1)*batch_size):((j)*batch_size)+mod(size(X,1),nbatches);
        else
            range=((1+(j-1)*batch_size):((j)*batch_size));
        end
        X_batch=X(range,:);
        y_batch=y(range,:);
        x_h_iter_hold= X_batch*theta;
        x_h_iter_hold = soft_max(x_h_iter_hold);
        x_h = vertcat(x_h,x_h_iter_hold);
        grad = ((1/num_classes)*(X_batch'*(one_hot_encoded(range,:)-x_h_iter_hold)));
        theta = theta + lr*grad;
    end
    loss = cost_cross_entropy(x_h,one_hot_encoded);
    losses(i) = loss;
    x_h_t = X_t*theta;
    x_h_t = soft_max(x_h_t);
    loss_test = cost_cross_entropy(x_h_t,one_hot_encoded_t);
    losses_test(i) = loss_test; 
    if best_train_loss > loss
        best_train_loss = loss;
    end
    if best_test_loss > loss_test
        best_test_loss = loss_test;
        best_epoch_weigths = theta;
    end
end
plot(losses);
%% predict old and new data
prediction_train = X*best_epoch_weigths;
prediction_train = soft_max(prediction_train);
[~,prediction_train] = max(prediction_train,[],2);
prediction_train=prediction_train-1;

prediction_test = X_t*best_epoch_weigths;
prediction_test = soft_max(prediction_test);
[~,prediction_test] = max(prediction_test,[],2);
prediction_test=prediction_test-1;



%% get confusion matrix
cm_test = confusionmat(y_t,prediction_test);

cm_train = confusionmat(y,prediction_train);

%stats computations

train_stats=compute_stats(cm_train);
test_stats=compute_stats(cm_test);
%% plot losses
plot(losses)
hold on 
plot(losses_test)
hold off

%% plot weights images

w_i=weights_to_images( theta,28,28,10);