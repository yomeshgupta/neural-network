## neural-network
> Neural Network implementation in python.

## Usecase

This repo contains python implementation of Multi-class Classification using Neural Network. Currently, the network is trained on E-commerce data where given a certain features the network classifies a user into one of the four predefined classes. Dataset is provided.

Features :
- Is_mobile(0/1)
- N_products_viewed (int >= 0)
- Visit_duration (real >= 0)
- Is_returning_visitor (0/1)
- Time_of_day (0 / 1 / 2 / 3 = 24h split into 4 categories)

Output :
- User_action (bounce (1), add_to_cart (2), begin_checkout (3), finish_checkout (4))

By giving different dataset or modifying the script, you can predict anything you want. The dataset is normalised while running. If dataset is already normalised then that is not required.

## Requirements

- [Numpy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Scipy](https://www.scipy.org/)
- [Sklearn](http://scikit-learn.org/)

## Installation

Just clone this repo and do :

    python main.py