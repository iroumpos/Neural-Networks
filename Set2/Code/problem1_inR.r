# Conjugate Gradient method
# 
# Requires an initial point x - matrix nx1
# Requires a function parameters to the gradient and hessian matrices of the function
# Requires a maximum number of iterations, used to stop if algorithm diverges
# Requires a tolerance for exiting condition
# Requires a mask boolean for whether or not to print out values inbetween iterations
# Requires a value ranging from 1,2,3 for the type of beta method 

conjuage_gradient = function(init_point, gradient, hessian, maxIter, tol, mask, betaMethod) {
  
  x = init_point
  
  for(i in 1:maxIter) {
    
    if(!mask) { # if the user does not want print out statements
      print(sprintf("      Iteration: %d", i))
      print("x value:")
      print(x)
    }
    
    g = gradient(x) # find gradient at x
    
    if(!mask) { # if the user does not want print out statements
      print("gradient value:")
      print(g)
    }
    
    # calculate the frobenius norm of the gradient vector evaluated at x
    # remember, stationary points occur when gradient is zero, so we take
    # the square root of the sum of squares of the gradient values and
    # compare to the tolerance
    if(norm(g, "F") < tol) { # convert gradient into single value 
      # through frobenius norm
      print("Stationary Point Found at point:")
      print(x)
      print(sprintf("Iterations taken: %d", i))
      return(x)
    }
    
    # creating the search direction vectors
    if(i==1) { # first search direction
      p = -g
    }
    else { # use beta
      if(betaMethod == 1) { # Common choice
        beta = (t(g-g0)%*%g)/(t(g0)%*%p)
      }
      else if (betaMethod == 2) { # hestens and stiefel
        beta = (t(g)%*%g)/(t(g0)%*%g)
      }
      else { # Fletcher and Reeves
        beta = (t(g-g0)%*%g)/(t(g0)%*%g0)
      }
      
      if(!mask) { # if the user does not want print out statements
        print("beta value:")
        print(beta)
      }
      
      p = -g+beta[1]*p
    }
    
    
    alpha = -(t(g)%*%p) / (t(p)%*%hessian(x)%*%p) # learning rate
    print("alpha value:")
    print(alpha)
    x = x - alpha[1]*g # find next x
    
    g0 = g # update previous gradient to current
  }
  
  print("MAXIMUM ITERATIONS REACHED")
  print("Current point is: ")
  print(x)
  print("Error: ")
  print(abs(norm(g, "F")-tol)) # computing the error 
  return(x)
}

# example function:
# (x+y)^4-12xy+x+y-1
# Has three stationary points
# 1) [-0.65, -0.65]
# 2) [0.5654, 0.564]
# 3) [0.0849, 0.0849]

gradient = function(X) {
  x = X[1,1]
  y = X[2,1]
  g1 = 2.5*x + y + 2*(0.5*x + y)**3
  g2 = x + 4*y + 4*(0.5*x + y)**3
  matrix(c(g1, g2), ncol=1)
}

hessian = function(X) { 
  x = X[1,1]
  y = X[2,1]
  fxx = 2.5 + 3*(0.5*x + y)**2
  fxy = 1 + 6*(0.5*x + y)**2
  fyy = 4 + 12*(0.5*x + y)**2
  matrix(c(fxx, fxy, fxy, fyy), ncol=2)
}

# initial points
p1 = matrix(c(3, 3), ncol=1)

# Function call
min = conjuage_gradient(p1,gradient, hessian, 5, 1e-10, 0,3)


# Steepest Descent
# 
# Requires an initial point x - matrix nx1
# Requires a function parameters to the gradient and hessian matrices of the function
# Requires a maximum number of iterations, used to stop if algorithm diverges
# Requires a tolerance for exiting condition
# Requires a mask boolean for whether or not to print out values inbetween iterations

steepest_descent = function(init_point, gradient, hessian, maxIter, tol, mask) {
  
  x = init_point
  
  for(i in 1:maxIter) {
    
    if(!mask) { # if the user does not want print out statements
      print(sprintf("      Iteration: %d", i))
      print("x value:")
      print(x)
    }
    
    g = gradient(x) # find gradient at x
    
    if(!mask) { # if the user does not want print out statements
      print("gradient value:")
      print(g)
    }
    
    # calculate the frobenius norm of the gradient vector evaluated at x
    # remember, stationary points occur when gradient is zero, so we take
    # the square root of the sum of squares of the gradient values and
    # compare to the tolerance
    if(norm(g, "F") < tol) { # convert gradient into single value 
      # through frobenius norm
      print("Stationary Point Found at point:")
      print(x)
      print(sprintf("Iterations taken: %d", i))
      return(x)
    }
    
    # creating the search direction vectors
    p = -g
    
    # Here we have a dynamic learning rate that is minimized against
    # the line of the next iteration
    alpha = -(t(g)%*%p) / (t(p)%*%hessian(x)%*%p) 
    
    x = x - alpha[1]*g # find next x
    
    
  }
  
  print("MAXIMUM ITERATIONS REACHED")
  print("Current point is: ")
  print(x)
  print("Error: ")
  print(abs(norm(g, "F")-tol)) # computing the error 
  return(x)
}


# example function:
# (x+y)^4-12xy+x+y-1
# Has three stationary points
# 1) [-0.65, -0.65]
# 2) [0.5654, 0.564]
# 3) [0.0849, 0.0849]

gradient = function(X) {
  x = X[1,1]
  y = X[2,1]
  g1 = 2.5*x + y + 2*(0.5*x + y)**3
  g2 = x + 4*y + 4*(0.5*x + y)**3
  matrix(c(g1, g2), ncol=1)
}

hessian = function(X) { 
  x = X[1,1]
  y = X[2,1]
  fxx = 2.5 + 3*(0.5*x + y)**2
  fxy = 1 + 6*(0.5*x + y)**2
  fyy = 4 + 12*(0.5*x + y)**2
  matrix(c(fxx, fxy, fxy, fyy), ncol=2)
}


# initial points
p1 = matrix(c(3, 3), ncol=1)

# Function call
min = steepest_descent(p1,gradient, hessian, 10, 1e-10, 0)

