function [o_h_e] = one_hot_encoding(y,n_class)
%ONE_HOT_ENCODING Summary of this function goes here
%   Detailed explanation goes here
    n_records = size(y,1);
    o_h_e = zeros(n_records,n_class);
    for i=1:n_records
        o_h_e(i,y(i)+1)=1;
    end
end

