function [smax] = soft_max(x_h)
%SOFT_MAX Summary of this function goes here
    for i=1:size(x_h,2)
        smax(:,i)=exp(x_h(:,i))./(sum(exp(x_h),2));
        smax(isnan(smax))=0;
    end
end

