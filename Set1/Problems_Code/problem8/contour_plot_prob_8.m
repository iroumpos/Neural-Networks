nx = 200;
ny = 200;

x = linspace(-50, 60, nx);
y = linspace(-50, 60, ny);
f = zeros(nx, ny);

for i = 1:nx
    for j = 1:ny
        f(i, j) = obj([x(i), y(j)]);
    end
end

figure;
%contour_levels = linspace(min(f(:)), max(f(:)), 21);

%contourf(x, y, f, contour_levels); % Filled contour plot
contour(x, y, f, 21);
colorbar;
hold on;

% Example: Add a point at x=2, y=3
x_point = 4.237;
y_point = 4.7185;
f_point = obj([x_point, y_point]);

plot(x_point, y_point, 'ro', 'MarkerSize', 10, 'LineWidth', 2); % Red point
text(x_point, y_point, ['  (' num2str(x_point) ',' num2str(y_point) ',' num2str(f_point) ')'], 'Color', 'r', 'FontSize', 10);
hold off;

xlabel('w_{1,1}')
ylabel('w_{1,2}')
title("Contour Plot of MSE performance index");


function [f] = obj(x)
    f = 676 - 176.8 * x(1) - 124.8 * x(2) + 12.4 * x(1)^2 + 6.4 * x(2)^2 + 15.2 * x(1) * x(2);
end
