#! /usr/bin/python

# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

# The sinewave regression example

import logging
import sys
import argparse
import math
import random

import numpy as np

import mlp


def parseCmd(argv):
    parser = argparse.ArgumentParser(description="Application description")
    parser.add_argument("-l", "--logfile", help="Logile", default="log.log")
    args = parser.parse_args(argv[1:])
    return args


def initLogging(args):
    formatString = '[%(levelname)s][%(asctime)s] : %(message)s'
    # formatString = '[%(levelname)s][%(name)s] : %(message)s'
    logLevel = logging.INFO
    logging.basicConfig(format=formatString, level=logLevel, datefmt='%Y-%m-%d %I:%M:%S')


def getData1():
    x = np.linspace(0, 1, 40).reshape((40, 1))
    t = np.sin(2 * np.pi * x) + np.cos(4 * np.pi * x) + np.random.randn(40).reshape((40, 1)) * 0.2
    x = (x - 0.5) * 2
    return x, t


def getData2():
    x1 = np.linspace(0, 1, 50)
    x2 = np.linspace(0, 1, 50)
    x = []
    for i in x1:
        for j in x2:
            x.append([i, j])
    x = np.array(x)
    t = []
    for xx in x:
        t.append(math.cos(2.0 * np.pi * xx[0]) + math.cos(4.0 * np.pi * xx[1]) + (random.random() - 0.5) * 0.02)
    t = np.array(t)
    t = t.reshape((len(t), 1))
    return x, t


def main(argv):
    args = parseCmd(argv)
    initLogging(args)

    # Set up the data
    # x,t = getData1()
    x, t = getData2()

    # # Split into training, testing, and validation sets
    train = x[0::2, :]
    test = x[1::4, :]
    valid = x[3::4, :]
    traintarget = t[0::2, :]
    testtarget = t[1::4, :]
    validtarget = t[3::4, :]

    # Perform basic training with a small MLP
    net = mlp.mlp(train, traintarget, 20, outtype='linear')
    error = net.earlystopping(train, traintarget, valid, validtarget, 0.25)

    # show result for test values
    biasedTest = np.concatenate((test, -np.ones((np.shape(test)[0], 1))), axis=1)
    outputs = net.mlpfwd(biasedTest)

    print "Error: " + str(error)
    print "Test error: " + str((0.5 * sum((outputs - testtarget) ** 2))[0])

    with open("result.dat", "w") as f:
        for i in range(0, len(biasedTest)):
            f.write(str(biasedTest[i][0]) + " " + str(biasedTest[i][1]) + " " + str(outputs[i][0]) + "\n")

    with open("network_in.dat", "w") as f:
        for i, w1 in enumerate(net.weights1):
            for j, w in enumerate(w1):
                f.write(str(i) + " " + str(j) + " " + str(w) + "\n")
            f.write("\n")
    with open("network_hidden.dat", "w") as f:
        for w2 in net.weights2:
            f.write(str(w2[0]) + "\n")


    # # Plot the data
    # pl.plot(x,t,'o')
    # pl.xlabel('x')
    # pl.ylabel('t')
    # pl.plot(test,testtarget,'o')
    # pl.plot(test,outputs,'x')
    # pl.show()


    # # Test out different sizes of network
    # # count = 0
    # out = np.zeros((10,7))
    # for nnodes in [1,2,3,5,10,25,50]:
    #     for i in range(10):
    #         net = mlp.mlp(train,traintarget,nnodes,outtype='linear')
    #         out[i,count] = net.earlystopping(train,traintarget,valid,validtarget,0.25)
    #     count += 1

    # 	#
    # # print out
    # print out.mean(axis=0)
    # print out.var(axis=0)
    # print out.max(axis=0)
    # print out.min(axis=0)


if __name__ == "__main__":
    main(sys.argv)
