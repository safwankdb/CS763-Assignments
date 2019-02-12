function [ z ] = dist(model, data)

pad = ones(1, 3);
a = data(:, 1:3);
a = a./(a(:, 3) * pad);
hypothesis = a * model;
hypothesis = hypothesis./(hypothesis(:, 3) * pad);  
b = data(:, 4:6);
b = b./(b(:, 3) * pad);

x_pred = hypothesis(:, 1);
y_pred = hypothesis(:, 2);

x_true = b(:, 1);
y_true = b(:, 2);


squared_error = (x_pred - x_true).^2 + (y_pred - y_true).^2;
z = sqrt(squared_error);
end
