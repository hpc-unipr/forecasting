# Functions for theta forecasting - written by Dimitrios Thomakos, dimitrios.thomakos@gmail.com
#
#dataset with labelled columns
theta.forecasting <- function(dataset,M=5,has.dates=TRUE,lambda=1)
{

  #Save the variables' names  
  Lab <- labels(dataset)
  var.names <- NULL
  for (i in 1:length(Lab[[2]])){
	var.names <- c(var.names,Lab[[2]][i])
  }
  #Save the number of variables
  Nvar <- length(var.names)

  #Save number of observations
  Nobs <- length(dataset[,1])

  #Transform the dataset in a matrix
  X <- as.matrix(dataset)

  
  # Compute sample means and trends
  dX  <- as.matrix(diff(X))  
  mdX <- apply(dX,2,mean) 
  maX <- apply(X,2,function(zzz) {as.matrix(filter(zzz,rep(1,M)/M,method="convolution",sides=1L)) } )  
  # Fill the MA values
  maX[1:(M-1),] <- X[1:(M-1),]
  rownames(maX) <- rownames(X)
  mmX <- apply(diff(maX),2,mean)  
  ctX <- as.matrix(apply(X,1,mean))  
  mcX <- mean(diff(ctX))
  
  # The naive, naive with drift forecasts and the SMA forecast
  # Add sample mean
  f00 <- apply(X,2,mean)
  f01 <- X[Nobs,] 
  f02 <- mdX + f01  
  f03 <- maX[Nobs,]
  # Add linear trend
  lt.out <- lm(X[,1]~seq(Nobs))  
  f04 <- coefficients(lt.out)[1] + coefficients(lt.out)[2]*(Nobs+1)  
  
  # OK, now compute differences, detrending etc. and the associated forecasts
  #
  # A simple function to get the optimal theta for the standard case
  .opt.theta <- function(.dX) 
  {
    Nn <- NROW(.dX)
    mu <- mean(.dX)
    dY <- (.dX[seq(2,Nn,1)]-mu) 
    dZ <- (.dX[seq(1,Nn-1,1)]-mu)
    rho <- coefficients(lm(dY~dZ-1))  
  }
  # Get the univariate theta for the standard case
  theta.D <- apply(dX,2,.opt.theta)
  theta.MA<- 2*theta.D - 1
  # and compute the standard forecasts - univariate
  dTheta <- theta.D*dX[Nobs-1,] + (1-theta.D)*mdX
  # Adaptive standard theta
  dd <- abs(dX[Nobs-1,] - mdX)
  wd <- exp(-lambda*dd)/(1+exp(-lambda*dd))
  wTheta <- wd*dX[Nobs-1,] + (1-wd)*mdX
  Theta  <- theta.MA*X[Nobs,] + (1-theta.MA)*0.5*(X[Nobs,]+X[Nobs-1,])
  # Note that the forecasts below are based on reversing differencing
  f1 <- X[Nobs,] + dTheta
  f1a<- X[Nobs,] + wTheta
  f2 <- Theta + 0.5*mdX*(3-theta.MA)  
  #
  # Next, the trend stationary case - univariate
  #
  # Single Theta first
  eta.sma <- X - maX
  theta.SMA <- apply(eta.sma,2,.opt.theta)
  theta.BMA <- theta.SMA/(1 - apply(maX,2,var)/apply(eta.sma,2,var))
  theta.BMA[which(theta.BMA < 0)] <- theta.SMA[which(theta.BMA < 0)]
  f3 <- theta.SMA*X[Nobs,] + (1-theta.SMA)*maX[Nobs,] 
  f4 <- f3 + mmX
  f5 <- f3 + theta.SMA*mmX
  f6 <- theta.BMA*X[Nobs,] + (1-theta.BMA)*maX[Nobs,]
  f7 <- f6 + mmX
  f8 <- f6 + theta.BMA*mmX
  # 
  # and the double Theta, starting with a simple objective function 
  .opt2theta <- function(par,series,trend)
  {
    Q1 <- par[1]*series + (1-par[1])*trend
    Q2 <- par[1]*diff(series) + (1-par[1])*diff(trend)
    FQ <- Q1[-1] + par[2]*Q2 + (1-par[2])*mean(diff(trend),na.rm=TRUE) 
    uu <- series[-1]-FQ
    SS <- mean(uu^2,na.rm=TRUE)
    return(SS)
  }
  f9 <- NULL
  for (i in 1:Nvar){
	# call the optimizer
	out <- optim(c(0.5,0.5),.opt2theta,method="L-BFGS-B",series=X[-1,i],trend=maX[-Nobs,i],lower=c(0,0),upper=c(1,1)) 
	theta.DD1 <- out$par
	#
	Q1 <- theta.DD1[1]*X[-1,i] + (1-theta.DD1[1])*maX[-Nobs,i]
	Q2 <- theta.DD1[1]*diff(X[-1,i]) + (1-theta.DD1[1])*diff(maX[-Nobs,i])
	f9i <- Q1[-1] + theta.DD1[2]*Q2 + (1-theta.DD1[2])*mean(diff(maX[-Nobs,i]),na.rm=TRUE)
	f9i <- f9i[length(f9i)]
	#
	f9 <- c(f9,f9i)
	}

  
  # OK, now we repeat for the bivariate case
  #
  f10 <- f11 <- f12 <- f13 <- f14 <- f15 <- NA
  #

    # First the standard case
    tau <- seq(Nobs)
    # Plain in levels
	dim(X)
 	dim(matrix(mdX,nrow=Nobs,ncol=Nvar,byrow=TRUE)*tau)
    S <- X-matrix(mdX,nrow=Nobs,ncol=Nvar,byrow=TRUE)*tau
    Y <- S[seq(2,Nobs,1),]
    Z <- S[seq(1,Nobs-1,1),]
    Theta.L <- coefficients(lm(Y~Z-1))
    # Reduced rank in levels
    S11 <- crossprod(Z)  #crossprod(x,y) prodotto matriciale fra x e y
    S12 <- crossprod(Z,dX)
    S22 <- crossprod(dX)
    D11 <- eigen(solve(S11))
    sqrt.iS11 <- (D11$vectors)%*%sqrt(diag(D11$values))%*%(t(D11$vectors))   
    Dall <- eigen(sqrt.iS11%*%S12%*%S22%*%t(S12)%*%sqrt.iS11)
    Beta <- sqrt.iS11%*%(Dall$vectors[,1])
    Alpha<- solve(crossprod(Z%*%Beta))%*%crossprod(Z%*%Beta,dX)
    Theta.RR <- (Beta%*%Alpha+diag(rep(1,Nvar)))
    # Plain in differences
    dY <- (dX[seq(2,Nobs-1,1),]-matrix(mdX,nrow=Nobs-2,ncol=Nvar,byrow=TRUE))
    dZ <- (dX[seq(1,Nobs-2,1),]-matrix(mdX,nrow=Nobs-2,ncol=Nvar,byrow=TRUE))
    Theta.Diff <- coefficients(lm(dY~dZ-1))
    # Trend with common factor - careful with the singularity
    eta.CT <- X - ctX%*%matrix(1,nrow=1,ncol=Nvar)
    e2 <- eta.CT[-1,1]
    e1 <- eta.CT[-Nobs,1]
    Theta.CT <- coefficients(lm(e2~e1-1))
    # Trend with SMA
    eta.MA <- X - maX
    e2 <- eta.MA[-1,]
    e1 <- eta.MA[-Nobs,]
    Theta.MA <- coefficients(lm(e2~e1-1))
    # and the multivariate forecasts
    f10 <- mdX*(Nobs+1) + (X[Nobs,]-mdX*Nobs)%*%Theta.L
    f11 <- mdX*(Nobs+1) + (X[Nobs,]-mdX*Nobs)%*%Theta.RR
    f12 <- mdX + X[Nobs,] + (dX[Nobs-1,]-mdX)%*%Theta.Diff
    #
    f13 <- X[Nobs,]*Theta.CT + (ctX[Nobs,]%*%matrix(1,nrow=1,ncol=Nvar))*(1-Theta.CT) + mcX
    f14 <- X[Nobs,]%*%Theta.MA + maX[Nobs,]%*%(diag(rep(1,Nvar))-Theta.MA) 
    f15 <- f14 + mmX
  
  # Full return
  #if (is.null(x2))
  #{
  #  all.f <- c(f00,f01,f02,f03,f04,f1,f1a,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15) # fdo1,fdo2,fdo3,fdo4,
  #  names(all.f) <- c("Mean","Naive","Naive-D","SMA","LT","UR1","UR1-A","UR2","TS1","TS1-BC1","TS1-BC2","TS2","TS2-BC1","TS2-BC2","TS-DBL",
  #                    "BUR-L","BUR-RR","BUR-D","BCT","BMA","BMA-BC")
  #}
  #if (!is.null(x2))
  #{
  #  all.f <- rbind(f00,f01,f02,f03,f04,f1,f1a,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15)
  #  rownames(all.f) <- c("Mean","Naive","Naive-D","SMA","LT","UR1","UR1-A","UR2","TS1","TS1-BC1","TS1-BC2","TS2","TS2-BC1","TS2-BC2","TS-DBL",
  #                       "BUR-L","BUR-RR","BUR-D","BCT","BMA","BMA-BC")
  #}
  ## Return the forecasts
  #return(all.f)
  return(f12)
} 
