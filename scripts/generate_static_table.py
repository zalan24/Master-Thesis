import argparse
import csv
import os
import glob
import shutil
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import interpolate
import json

def process_benchmarks(sourcedir, dir):
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


    configs = []
    for filename in os.listdir(sourcedir):
        suffix = "_downsampled.csv"
        if filename.endswith(suffix):
            configname = filename.removesuffix(suffix)
            configFile = os.path.join(sourcedir, configname+'.json')
            data = genfromtxt(os.path.join(sourcedir, filename), delimiter=',', dtype=dtype)
            info = {}
            for entry in dtype[1:]:
                info[entry[0]] = np.average(data[entry[0]])
            with open(configFile) as f:
                conf = json.load(f)
                info['targetFps'] = ''
                if not conf['latencyReduction']:
                    info['latencyMode'] = 'none'
                elif conf['refreshMode'] == 'unlimited':
                    info['latencyMode'] = 'unlimited'
                elif conf['refreshMode'] == 'limited':
                    info['latencyMode'] = 'limited'
                    info['targetFps'] = conf['targetRefreshRate']
                elif conf['refreshMode'] == 'discretized':
                    info['latencyMode'] = 'vsync'
                    info['targetFps'] = conf['targetRefreshRate']
                else:
                    raise "Unknown latency mode: " + conf['refreshMode']
                info['latencyPool'] = conf['desiredSlop']
                if 'highstdDev' in configname:
                    info['noise'] = 'yes'
                else:
                    info['noise'] = 'no'
                if info['cpuWork'] < info['execWork']:
                    if info['execWork'] < info['deviceWork']:
                        info['bottleneck'] = 'exec+gpu'
                    else:
                        info['bottleneck'] = 'exec'
                else:
                    if info['cpuWork'] < info['deviceWork']:
                        info['bottleneck'] = 'gpu'
                    else:
                        info['bottleneck'] = 'cpu'
                info['name'] = configname
                if info['missRate'] < 0:
                    info['missRate'] = ''
            if info['latencyMode'] == 'limited':
                continue
            configs.append(info)

    configs.sort(key=lambda x: x['name'])

    propertiesToPlot = [
        ('latencyMode', 'Latency reduction mode'),
        ('noise', 'Noisy fps'),
        ('latencyPool', 'Latency pool'),
        ('targetFps', 'Target fps'),
        ('bottleneck', 'Bottleneck'),
        ('fps', 'Framerate'),
        ('latency', 'Latency'),
        ('missRate', 'Missed frames')]
        # ('name', 'Configname')

    fields = []
    for p in propertiesToPlot:
        fields.append(p[1])

    rows = []
    for info in configs:
        row = []
        for p in propertiesToPlot:
            row.append(info[p[0]])
        rows.append(row)

    filename = os.path.join(dir, 'static_benchmarks.csv')
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, dialect='unix')
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # add arguments to the parser
    parser.add_argument("outdir")
    parser.add_argument("sourcedir")

    # parse the arguments
    args = parser.parse_args()
    process_benchmarks(args.sourcedir, args.outdir)