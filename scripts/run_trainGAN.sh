#!/bin/bash
for MSE_LAMBDA in 10 20 50 100;
do python ../trainGAN.py --mse_lambda $MSE_LAMBDA;
done
