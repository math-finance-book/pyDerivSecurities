import numpy as np
from scipy.stats import norm
import scipy.optimize as optimize
from math import erf, sqrt
#import math
#from math import pow, exp, sqrt
#from scipy import stats


def disc_function(FV,r,T):
    PV = FV * np.exp(-r*T)
    return PV

def bs_d1_d2(St,K,r,q,sigma,t):
    d1 = np.log(St/K)
    d1 += (r-q + sigma**2/2) * t
    #with np.errstate(divide='ignore'):
    #    d1 /= sig * t**0.5
    if sigma*t == 0:
        d1 = np.inf*np.sign(d1)
    else:
        d1 = np.divide(d1,sigma * np.sqrt(t))
    d2 = d1 - sigma*np.sqrt(t)
    #print(d1,d2)
    return d1,d2

def cdf_approx(dn,call):
    if call:
        Ndn = (0.50 * (1.0 + erf(dn / sqrt(2.0))))
    else:
        Ndn = (0.50 * (1.0 + erf(-dn / sqrt(2.0))))
    return Ndn

def Nd(d,call):
    if call:
        Nd = norm.cdf(d)
    else:
        Nd = norm.cdf(-d)
    return Nd

#def bs_delta(d1,d2,call):
#    Nd1 = cdf_approx(dn=d1,call=call)
#    Nd2 = cdf_approx(dn=d2,call=call)
#    return Nd1,Nd2

def bs_delta(St,K,r,q,sigma,t,call):
    d1,_ = bs_d1_d2(St,K,r,q,sigma,t)
    Nd1 = Nd(d1,call)
    return Nd1*np.exp(-q*t)

def bs_gamma(d1,St,sigma,t,q):
    gamma = norm.pdf(d1)
    #with np.errstate(divide='ignore'):
    #    gamma /= (St*sig*np.sqrt(t)) 
    gamma=np.divide(gamma,St*sigma*np.sqrt(t))*np.exp(-q*t)
    return gamma

def bs_price(St,K,r,q,sigma,t,call):
    '''Calculate the price of a European call/put option using Black-Scholes formula.

    Args:
        St (num): current price of underlying asset.
        K (num): strike price.
        r (num): risk-free rate.
        q (num): dividend yield
        sigma (num): volatility of the underlying asset
        t (num): time-to-maturity
        call (bool): True returns call price, False returns put price

    Returns:
        price (num): price of the option'''
    pvk = disc_function(K,r,t)
    d1,d2=bs_d1_d2(St,K,r,q,sigma,t)
    Nd1,Nd2=Nd(d1,call),Nd(d2,call)
    if call:
        price = np.exp(-q*t)*St*Nd1 - pvk*Nd2
    else:
        price = pvk*Nd2 - np.exp(-q*t)*St*Nd1 
    return price


def bs_implied_vol(S,K,r,q,T,Price,call=True,guess=0.2,tol=1e-6):
    """    
    Inputs:
    S = initial stock price
    K = strike price
    r = risk-free rate
    q = dividend yield
    T = time to maturity
    CallPrice = call price
    """
    def objective(sigma):
        return bs_price(S,K,r,q,sigma,T,call) - Price 
    return optimize.root_scalar(objective,x0=guess,xtol=tol,method='bisect',bracket=(0,100)).root

bs_d1_d2=np.vectorize(bs_d1_d2)
Nd=np.vectorize(Nd)
bs_price=np.vectorize(bs_price)
bs_implied_vol=np.vectorize(bs_implied_vol)
############################################################
############################################################

def blackscholes(S0,K,r,q,sig,T,call=True):
    '''Calculate option price using B-S formula.

    Args:
        S0 (num): initial price of underlying asset.
        K (num): strick price.
        r (num): risk free rate.
        q (num): dividend yield
        sig (num): Black-Scholes volatility.
        T (num): maturity.
        call (bool): True returns call price, False returns put price.

    Returns:
        num
    '''
    if sig*T == 0:
        return np.maximum(0, np.exp(-q*T)* S0-np.exp(-r*T)*K) if call else  np.maximum(0, np.exp(-r*T)*K-np.exp(-q*T)*S0)

    d1 = (np.log(S0/K) + (r -q + sig**2/2) * T)/(sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    #print(d1,d2)

    if call:
        return np.exp(-q*T)*S0 * norm.cdf(d1,0,1) - K * np.exp(-r * T) * norm.cdf(d2,0, 1)
    else:
        return np.exp(-q*T)*S0 * -norm.cdf(-d1,0,1) + K * np.exp(-r * T) * norm.cdf(-d2,0, 1)
    

#blackscholes=np.vectorize(blackscholes)

def blackscholes_delta(S0, K, r, q, sig, T, call = True):
    '''Calculate option price using B-S formula.

    Args:
        S0 (num): initial price of underlying asset.
        K (num): strick price.
        r (num): risk free rate.
        q (num): dividend yield
        sig (num): Black-Scholes volatility.
        T (num): maturity.
        call (bool): True returns call price, False returns put price.

    Returns:
        num
    '''
    d1 = (np.log(S0/K) + (r -q + sig**2/2) * T)/(sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
#     norm = sp.stats.norm
    if type(call) == bool:
        if call:
            return np.exp(-q*T)*norm.cdf(d1,0,1)
        else:
            return np.exp(-q*T)*norm.cdf(-d1,0,1)
    else:
        print("Not a valid value for call")


def black_scholes_call(S, K, r, sigma, q, T):
    """
    Inputs:
    S = initial stock price
    K = strike price
    r = risk-free rate
    sigma = volatility
    q = dividend yield
    T = time to maturity
    """
    if sigma == 0:
        return max(0, np.exp(-q * T) * S - np.exp(-r * T) * K)
    else:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        N1 = norm.cdf(d1)
        N2 = norm.cdf(d2)
        return np.exp(-q * T) * S * N1 - np.exp(-r * T) * K * N2

def black_scholes_put(S, K, r, sigma, q, T):
    """
    Inputs:
    S = initial stock price
    K = strike price
    r = risk-free rate
    sigma = volatility
    q = dividend yield
    T = time to maturity
    """
    if sigma == 0:
        return max(0, np.exp(-r * T) * K - np.exp(-q * T) * S)
    else:
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        N1 = norm.cdf(-d1)
        N2 = norm.cdf(-d2)
        return np.exp(-r * T) * K * N2 - np.exp(-q * T) * S * N1

def black_scholes_call_delta(S, K, r, sigma, q, T):
    """
    Inputs:
    S = initial stock price
    K = strike price
    r = risk-free rate
    sigma = volatility
    q = dividend yield
    T = time to maturity
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.cdf(d1)

def black_scholes_call_gamma(S, K, r, sigma, q, T):
    """
    Inputs:
    S = initial stock price
    K = strike price
    r = risk-free rate
    sigma = volatility
    q = dividend yield
    T = time to maturity
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    nd1 = np.exp(-d1 ** 2 / 2) / np.sqrt(2 * np.pi)
    return np.exp(-q * T) * nd1 / (S * sigma * np.sqrt(T))

def black_scholes_call_implied_vol(S, K, r, q, T, CallPrice,guess=0.2,tol=1e-6):
    """    
    Inputs:
    S = initial stock price
    K = strike price
    r = risk-free rate
    q = dividend yield
    T = time to maturity
    CallPrice = call price
    """
    def objective(sigma):
        return black_scholes_call(S, K, r, sigma, q, T) - CallPrice
    return optimize.root_scalar(objective,x0=guess,xtol=tol,method='bisect',bracket=(0,100)).root


blackscholes=np.vectorize(blackscholes)
blackscholes_delta=np.vectorize(blackscholes_delta)
black_scholes_call=np.vectorize(black_scholes_call)
black_scholes_put=np.vectorize(black_scholes_put)
black_scholes_call_delta=np.vectorize(black_scholes_call_delta)
black_scholes_call_gamma=np.vectorize(black_scholes_call_gamma)
black_scholes_call_implied_vol=np.vectorize(black_scholes_call_implied_vol)






