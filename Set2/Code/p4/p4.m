function p4(neurons, learning_rate)


% Validate input arguments
if nargin < 2
    error('Please provide the number of neurons and learning rate as input arguments.');
end

% Initialize data
W1 = rand(neurons, 1) - 0.5;
W2 = rand(1, neurons) - 0.5;
b1 = rand(neurons, 1) - 0.5;
b2 = rand - 0.5;
a1 = zeros(neurons, 1);

% Output the initial set
W1_0 = W1;
b1_0 = b1;
W2_0 = W2;
b2_0 = b2;

alfa = learning_rate;  % learning rate
tol = 0.001;           % tolerance
mse = 1;               % mean square error
iter = 0;
max_iter = 10000;
figure;

while (mse > tol && iter < max_iter)
    mse = 0;
    i = 0;
    iter = iter + 1;
    
    for P = -2 : 0.1 : 2
        i = i + 1;
        T = 1 + sin(3*pi*P/8);
        
        a1 = logsig(W1 * P + b1);
        a2 = max(0, W2 * a1 + b2);
        
        mse = mse + (T - a2)^2;
        A(i) = a2;
        
        dlogsig = diag((1 - a1) .* a1);
        s2 = -2 * (T - a2);
        s1 = dlogsig * W2' * s2;
        
        W2 = W2 - alfa * s2 * a1';
        W1 = W1 - alfa * s1 * P;
        b2 = b2 - alfa * s2;
        b1 = b1 - alfa * s1;
    end
    
    P = -2 : 0.1 : 2;
    
    if (mod(iter, 10) == 0)
        plot(P, A, 'g:')
    end
    
    hold on;
end

% Display in graph
P = -2 : 0.1 : 2;
T = 1 + sin(3*pi*P/8);
plot(P, T, 'r-', P, A, 'b+')
title(['Number of neurons: ' num2str(neurons) ', learning rate = ' num2str(learning_rate)]);
text(-1.8, 1.7, 'red ---- original function');
text(-1.8, 1.6, 'blue ---- approximation');
text(-1.8, 1.5, 'green ---- intermediate results');
xlabel('P'), ylabel('Target vs. output');
W1
b1
W2
b2
iter
