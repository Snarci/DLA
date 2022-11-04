function [out_images] = weights_to_images(theta,img_w,img_h,n_classes)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    out_images=reshape(theta,[img_w,img_h,n_classes]);
    for i=1:n_classes
        x=floor(n_classes/5)+1;
        y=5;
        %disp(strcat(string(x)," aaaa ",string(y)));
        subplot(x,y,i);
        imshow(out_images(:,:,i));
        title(strcat("Class n: ",num2str(i-1)));
    end
end

