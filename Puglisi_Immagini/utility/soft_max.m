function [smax] = soft_max(x_h)
%SOFT_MAX Summary of this function goes here
    [n,~] = max(x_h,[],2);
    smax=exp(x_h-n)./(sum(exp(x_h-n),2));

end

%old
%    smax = x_h;
%    for i=1:size(x_h,1)
%        [n,~]=max(x_h(i,:));
%        for j=1:size(x_h,2)
%            smax(i,j)=exp(x_h(i,j)-n)/(sum(exp(x_h(i,:)-n)));
%        end
%        
%   end