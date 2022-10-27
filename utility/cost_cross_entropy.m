function [cost] = cost_cross_entropy(x_h,one_hot_encoding)
%COST_FUNCTION Summary of this function goes here

    cost=(1/size(x_h,1))*sum(-log(sum(x_h .* one_hot_encoding,2)+eps));
end

