import argparse
import csv
import os
import glob
import shutil
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy import interpolate

def process_benchmarks(dir, sources):
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

    for s in sources:
        data = genfromtxt(s, delimiter=',', dtype=dtype)
        label = "???"
        if "control" in s:
            label = "control"
        elif "unlimited" in s:
            label = "unlimited"
        elif "limited" in s:
            label = "limited"
        elif "vsync" in s:
            label = "vsync"
        configs.append((label, data))

    plt.title("Fps")
    plt.ylabel("fps")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['fps'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_fps.png'))
    plt.clf()

    plt.title("Latency")
    plt.ylabel("ms")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['latency'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_latency.png'))
    plt.clf()

    plt.title("Slop")
    plt.ylabel("ms")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['slop'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_slop.png'))
    plt.clf()

    plt.title("Delay")
    plt.ylabel("ms")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['execDelay'] + data['deviceDelay'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_delay.png'))
    plt.clf()

    plt.title("Potential fps")
    plt.ylabel("ms")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['potentialFps'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_potential_fps.png'))
    plt.clf()

    plt.title("Potential cpu fps")
    plt.ylabel("ms")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['potentialCpuFps'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_potential_cpu_fps.png'))
    plt.clf()

    plt.title("Potential execution fps")
    plt.ylabel("ms")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['potentialExecFps'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_potential_exec_fps.png'))
    plt.clf()

    plt.title("Potential gpu fps")
    plt.ylabel("ms")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['potentialDeviceFps'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_potential_gpu_fps.png'))
    plt.clf()

    plt.title("Work time")
    plt.ylabel("ms")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['workTime'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_work.png'))
    plt.clf()

    plt.title("Cpu work time")
    plt.ylabel("ms")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['cpuWork'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_cpu_work.png'))
    plt.clf()

    plt.title("Execution work time")
    plt.ylabel("ms")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['execWork'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_exec_work.png'))
    plt.clf()

    plt.title("Device work time")
    plt.ylabel("ms")
    plt.xlabel("period")
    for c in configs:
        name, data = c
        plt.plot(data['period'], data['deviceWork'], '-', label=name)
    plt.legend()
    plt.savefig(os.path.join(dir, 'combined_device_work.png'))
    plt.clf()


if __name__=="__main__":
    parser = argparse.ArgumentParser()

    # add arguments to the parser
    parser.add_argument("outdir")
    parser.add_argument('files', type=str, nargs='+', help='Downsampled csv files')

    # parse the arguments
    args = parser.parse_args()
    process_benchmarks(args.outdir, args.files)