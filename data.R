n = 100
p = 500
sigma_noise = 0.5
beta = rep(0, p)
beta[1:6] = c(5,10,3,80,90,10)
XData = matrix(rnorm(n*p,sd=10), nrow = n, ncol=p)
X=XData
for (i in 1:500){
  X[,i]=((X[,i]-sum(X[,i])/n))/((var(X[,i])*99/100)^0.5)
}
YData = X %*% beta + rnorm(n, sd = sigma_noise)
YData=YData-sum(YData)/100
XData=X