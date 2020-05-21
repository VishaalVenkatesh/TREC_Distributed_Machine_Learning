#I am going to use squared loss i.e. Loss=(y - h(x))^2 where h(x)=WX 
#You could also do a slight variant where you have
#a bias term in the hypothesis i.e. h(x)=WX+b

#Estimate the gradient as follows
grad<-function(x,y,w)
{
 gradientPt<--2*x*(y-x*w)
 gradient<-sum(gradientPt)
 return(gradient)
}

#Perform the descent
grad.descent<-function(X,Y,maxitr)
{
#Keep track of the weight in each iteration
#wItr<-matrix(data=1,nrow=maxitr,ncol=1)	
#Set the number of examples
nEG<-nrow(as.matrix(X))	
#Set the alpha parameter
alpha=0.5
#Set the number of iterations 
for (i in 1:maxitr)
{
 W <- W-(alpha*grad(X,Y,W))/nEG
 print(W)
}
return(W)
}

#Perform Stochastic Gradient Descent
sgd.descent<-function(X,Y,W,maxitr)
{
#Set the alpha parameter
alpha=0.5
#Set the number of iterations 
for (i in 1:maxitr)
{
	#pick an example at random
	rnd<-sample(x=1:20,size=1)
	#estimate the gradient at this random point
	rndGrd<-grad(X[rnd],Y[rnd],W)
	W<-W-alpha*rndGrd
	print(W)
}
return(W)
}

#Define the hypothesis
h<-function(X,W)
{return (X*W)}

#Find the loss
Loss<-matrix(data=0,nrow=20,ncol=1)
W<-grad.descent(X,Y,1,10)
for(j in 1:20)
{
 Loss[j]<-(Y[j]-h(X[j],W))^2
}
# Find the mean of the Squared Loss
mean(Loss)