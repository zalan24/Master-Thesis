import argparse
import csv
import os
import glob
import shutil
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import interpolate

RESOLUTION = 100

def process_benchmark(source, dir):
    dtype = [
        ('period', float),
        ('fps', float),
        ('latency', float),
        ('slop', float),
        ('cpuWork', float),
        ('execWork', float),
        ('deviceWork', float),
        ('workTime', float),
        ('missRate', float)]
    os.makedirs(dir, exist_ok=True)

    files = glob.glob(os.path.join(dir, '*'))
    for f in files:
        os.remove(f)

    shutil.copyfile(source, os.path.join(dir, 'source.csv'))

    data = genfromtxt(source, delimiter=',', dtype=dtype)
    # np.sort(data, order='period')

    # print(f'len: {data.size}')
    # len = 

    subarray = data[0:10]
    # print(subarray['fps'])
    # for row in data:
        # print(f'aoeu {row}')

    ind = 0
    lastInd = 0
    lastP = 0
    numSamples = 0
    for row in data:
        p = row['period'] % 1
        # print(f'aoeu {row} {p} {lastP}')
        if p < lastP:
            # New period begins
            numSamples = numSamples+1
            subarray = data[lastInd:(ind+1)]
            lastInd = ind

            period = subarray['period'] % 1
            for entry in dtype[1:]:
                interpValues = interpolate.interp1d(period, subarray[entry[0]])
                # TODO linear space resolution...
                # linearValues = interpValues(numpy.linspace(0, 1, num=))
                # TODO can I just use t=period????
                # TODO create a window
                # resampled = scipy.signal.resample(linearValues, RESOLUTION, window=)
                # TODO sum these

            print(f'aoeu {row}')
        lastP = p
        ind = ind+1

    # TODO calc avg from summed samples
    # TODO plot stuff

    # flinear = interpolate.interp1d(x, y)
    # fcubic = interpolate.interp1d(x, y, kind='cubic')

    # xnew = np.arange(0.001, 20, 1)
    # ylinear = flinear(xnew)
    # ycubic = fcubic(xnew)
    # plt.plot(x, y, 'X', xnew, ylinear, 'x', xnew, ycubic, 'o')
    # plt.show()

    # x = np.arange(1,11)
    # y = 2 * x + 5
    # plt.title("Matplotlib demo")
    # plt.xlabel("x axis caption")
    # plt.ylabel("y axis caption")
    # plt.plot(x,y)
    # plt.savefig(os.path.join(dir, 'test.png'))

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # add arguments to the parser
    parser.add_argument("source")
    parser.add_argument("outdir")

    # parse the arguments
    args = parser.parse_args()
    process_benchmark(args.source, args.outdir)