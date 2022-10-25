%% load files
load('C:\Users\39342\Desktop\DLA\MNIST_dataset\MNIST_train_labels.mat')
load('C:\Users\39342\Desktop\DLA\MNIST_dataset\MNIST_train_images.mat')
load('C:\Users\39342\Desktop\DLA\MNIST_dataset\MNIST_test_labels.mat')
load('C:\Users\39342\Desktop\DLA\MNIST_dataset\MNIST_test_images.mat')

%% preprocess
[r,c,ch]=size(MNIST_train_images);
reshaped_train_img=reshape(MNIST_train_images,r*c,ch)';
[r,c,ch]=size(MNIST_test_images);
reshaped_test_img=reshape(MNIST_test_images,r*c,ch)';

%% normalize

reshaped_train_img = double(reshaped_train_img)/255;
reshaped_test_img = double(reshaped_test_img)/255;
%% extract labels
labels_out=[];
k = 3;
for i=1:size(reshaped_test_img,1)/100
    disp(i)
    dif= reshaped_train_img-reshaped_test_img(i,:);
    dif_sq = dif.^2;
    dif_sq_sum = sum(dif_sq,2);
    [C,I] = min(dif_sq_sum);
    current_label= MNIST_train_labels(I,:);
    labels_out=vertcat(labels_out,current_label);
end

%% visualize
for i=1:size(labels_out)
    imshow(MNIST_test_images(:,:,i));
    title(strcat("La label e' : ",string(labels_out(i,1))));
    pause(1);
end