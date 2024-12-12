qsmmd=function(I,V,alp)
{
  obj=function(x,y)
  {
    out=dchisq(y,V)*( 2*pnorm(x*sqrt( y / V )) -1 )^{I}	
    out
  }
  
  cdf=function(x)
  {
    integrate(obj,lower=0,upper=Inf,x=x)$value-alp	
  }
  
  round(uniroot(cdf,c(0,100))$root,2)  
}  

