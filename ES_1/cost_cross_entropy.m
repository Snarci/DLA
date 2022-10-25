function [cost] = cost_cross_entropy(x,y,theta)
%COST_FUNCTION Summary of this function goes here
h = (x*theta);
cost = (1/size(x,1))*sum(log(exp(y)./(sum(exp(h),2))));
end

