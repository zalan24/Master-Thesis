import argparse
import csv
import os
import glob
import shutil
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import interpolate

RESOLUTION = 128

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
        ('missRate', float),
        ('execDelay', float),
        ('deviceDelay', float),
        ('potentialFps', float),
        ('potentialCpuFps', float),
        ('potentialExecFps', float),
        ('potentialDeviceFps', float)]
    os.makedirs(dir, exist_ok=True)

    files = glob.glob(os.path.join(dir, '*'))
    for f in files:
        os.remove(f)

    shutil.copyfile(source, os.path.join(dir, 'source.csv'))

    data = genfromtxt(source, delimiter=',', dtype=dtype)
    downsamplePoints = np.linspace(0, 1, num=RESOLUTION)
    result = np.zeros(RESOLUTION, dtype=dtype)
    result['period'] = downsamplePoints

    ind = 0
    lastInd = 0
    lastP = 0
    numSamples = 0
    for row in data:
        p = row['period'] % 1
        if p < lastP:
            # New period begins
            numSamples = numSamples+1
            subarray = data[lastInd:(ind+1)]
            lastInd = ind

            period = subarray['period'] % 1
            for entry in dtype[1:]:
                interpValues = interpolate.interp1d(period, subarray[entry[0]], assume_sorted=True, fill_value="extrapolate")
                minRes = period.size*4
                resMul = int(np.ceil(minRes/RESOLUTION))
                res = resMul * RESOLUTION
                samplePoints = np.linspace(0, 1, num=res)
                linearValues = interpValues(samplePoints)
                window = []
                for i in range(resMul):
                    window.append((i + 1) / resMul / resMul)
                for i in range(1, resMul):
                    window.append((resMul - i) / resMul / resMul)
                convolved = np.convolve(linearValues, window, mode='valid')
                lostItems = resMul-1
                # print(samplePoints[lostItems:-lostItems])
                blurredValues = interpolate.interp1d(samplePoints[lostItems:-lostItems], convolved, assume_sorted=True, fill_value="extrapolate")
                downsampled = blurredValues(downsamplePoints)
                result[entry[0]] = result[entry[0]] + downsampled

            # print(f'aoeu {row}')
        lastP = p
        ind = ind+1

    if numSamples == 0:
        return

    for entry in dtype[1:]:
        result[entry[0]] = result[entry[0]] / float(numSamples)

    np.savetxt(os.path.join(dir, 'downsampled.csv'), result, fmt='%.18e', delimiter=',', newline='\n')

    plt.title("Fps")
    plt.ylabel("fps")
    plt.xlabel("period")
    plt.plot(result['period'], result['fps'], '-', label="fps")
    plt.plot(result['period'], result['potentialFps'], '-', label="Potential fps")
    plt.plot(result['period'], result['potentialCpuFps'], '-', label="CPU fps")
    # plt.plot(result['period'], result['potentialExecFps'], '-', label="Exec fps")
    plt.plot(result['period'], result['potentialDeviceFps'], '-', label="GPU fps")
    plt.legend()
    plt.savefig(os.path.join(dir, 'fps.png'))
    plt.clf()

    plt.title("Work times")
    plt.ylabel("ms")
    plt.xlabel("period")
    plt.plot(result['period'], result['workTime'], '-', label="Async work")
    plt.plot(result['period'], result['cpuWork'], '-', label="CPU work")
    # plt.plot(result['period'], result['execWork'], '-', label="Execution work")
    plt.plot(result['period'], result['deviceWork'], '-', label="Device work")
    plt.legend()
    plt.savefig(os.path.join(dir, 'worktime.png'))
    plt.clf()

    plt.title("Latency & work time")
    plt.ylabel("ms")
    plt.xlabel("period")
    plt.plot(result['period'], result['workTime'], '-', label="Async work")
    plt.plot(result['period'], result['latency'], '-', label="Latency")
    plt.plot(result['period'], result['slop'], '-', label="Slop")
    plt.legend()
    plt.savefig(os.path.join(dir, 'latency_work.png'))
    plt.clf()

    plt.title("Latency & delay")
    plt.ylabel("ms")
    plt.xlabel("period")
    plt.plot(result['period'], result['latency'], '-', label="Latency")
    plt.plot(result['period'], result['slop'], '-', label="Slop")
    plt.plot(result['period'], result['execDelay'] + result['deviceDelay'], '-', label="Total delay")
    plt.legend()
    plt.savefig(os.path.join(dir, 'latency_delay.png'))
    plt.clf()

    plt.title("Bottleneck & delay")
    plt.ylabel("ms")
    plt.xlabel("period")
    plt.plot(result['period'], 1000 / result['potentialCpuFps'], '-', label="CPU work")
    plt.plot(result['period'], 1000 / result['potentialDeviceFps'], '-', label="GPU work")
    plt.plot(result['period'], result['execDelay'] + result['deviceDelay'], '-', label="Total delay")
    plt.legend()
    plt.savefig(os.path.join(dir, 'bottleneck_delay.png'))
    plt.clf()

    plt.title("Miss rate")
    plt.ylabel("fps & rate(%)")
    plt.xlabel("period")
    plt.plot(result['period'], result['fps'], '-', label="fps")
    plt.plot(result['period'], result['missRate'] * 100, '-', label="Miss rate")
    plt.legend()
    plt.savefig(os.path.join(dir, 'missrate.png'))
    plt.clf()


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # add arguments to the parser
    parser.add_argument("source")
    parser.add_argument("outdir")

    # parse the arguments
    args = parser.parse_args()
    process_benchmark(args.source, args.outdir)