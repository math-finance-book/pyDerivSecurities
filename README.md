# pyDerivSecurities

This Python library is an amalgamation of code presented in [Pricing and Hedging Derivative Securities: Theory and Methods](https://book.derivative-securities.org). Readers of the textbook, who prefer mastering the material in a hands-on, programmtic approach, may find this library hepful, especially when attempting the end-of-chapter exercises.

We offer not only access to the formulas and models presented in the book, but also many of the simulation techniques that are explained in the text.

# Installation

Users may run the following line in a command prompt to install the package in their preferred Python environment:

```
pip install git+https://github.com/math-finance-book/pyDerivSecurities
```

**Note:** users who wish to utilize this package in Google Colab can access this [tutorial](https://colab.research.google.com/drive/1qsl5pmzTNhIlLCDNGBQShsr8Mk_iMK1X?authuser=1#scrollTo=eex2mX90hFKZ) for help with installation in their Colab environment.


# Sample usage


```python
from pyDerivSecurities.formulas import bs_price
r = .05 #risk-free rate
q=0.02 #underlying dividend yield
sigma = .2 #underlying volatility
S0 = 50 #current price of underlying asset
K = 42 #strike price
T = 0.5 #time to maturity (years)
call=True
#calculate and display the value of a European-style call option using the Black-scholes formula
print(bs_price(S0,K,r,q,sigma,T,call))
```
```
8.80576090557119
```

# References

* Black, Fischer, and Myron Scholes. “The Pricing of Options and Corporate Liabilities.” Journal of Political Economy, vol. 81, no. 3, 1973, pp. 637–54. JSTOR, http://www.jstor.org/stable/1831029. Accessed 23 Jan. 2025.
* Reference 2.
* Reference 3.
* Reference 4.
