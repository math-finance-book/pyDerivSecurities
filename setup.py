from setuptools import setup, find_packages

setup(
    name="pyDerivSecurities",  
    version="0.1.0",  
    author="Kerry Back, Austin Clime, Hong Liu, Mark Loewenstein",  
    author_email="aclime@terpmail.umd.edu",  
    description="""Python code companion for 'Pricing and Hedging Derivative Securities: Theory and Methods'
                . This package contains code defined in the text and is intended to help readers complete end-of-chapter exercises 
                and experiment with models presented within the text.""",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/math-finance-book/pyDerivSecurities",  
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)