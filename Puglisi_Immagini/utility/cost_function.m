function [cost] = cost_function(x,y,theta)
%COST_FUNCTION Summary of this function goes here
h = sigmoid(x*theta);
cost = (-y'*log(h))+(1-y)'*log(1-h);
end

