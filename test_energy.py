import os
import time
import argparse
from psutil import Popen
from subprocess import DEVNULL

Python_Abs_Path = "/home/yzhang/miniconda3/envs/tsp/bin/python"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", "-n", default="default")
    args = parser.parse_args()
    out_file = f"energy_{args.name}.txt"

    print(f"out_file={os.path.abspath(out_file)}")

    # run powerjoular
    cmd_powerjoular = f"powerjoular -t > {out_file}"
    p_powerjoular = Popen(cmd_powerjoular, shell=True).children()[0]

    # wait for powerjoular to start
    print("waiting for powerjoular to start")
    time.sleep(5)

    # run testing script
    # test_script = "/home/yzhang/tsp/benchmarking-tsp/main.py"
    test_script = "/home/yzhang/tsp/benchmarking-tsp/dummy.py"

    cmd = f"{Python_Abs_Path} {test_script}"
    print(f"monitoring '{cmd}'")

    t1 = time.time()

    p_tsp_solver = Popen(cmd, shell=True, stdout=DEVNULL)
    p_tsp_solver.communicate()

    t2 = time.time()
    duration = t2 - t1
    print(f"{duration=:.2f}s")

    # once testing is done, terminate powerjoular
    cmd_terminate = f"kill -2 {p_powerjoular.pid}"
    print(cmd_terminate)
    Popen(cmd_terminate, shell=True)

    # make sure the out_file is closed
    time.sleep(0.1)

    # read out_file and calculate avg power
    with open(out_file) as r:
        lines = r.read().strip().split("\n")
    num_records = 0
    for l in lines:
        if l.startswith("CPU: "):
            num_records += 1
        if l.startswith("Total energy: "):
            total_energy = float(l.split(":")[1].split("Joules")[0].strip())
        if l.startswith("	CPU energy: "):
            cpu_energy = float(l.split(":")[1].split("Joules")[0].strip())
        if l.startswith("	GPU energy: "):
            gpu_energy = float(l.split(":")[1].split("Joules")[0].strip())
    avg_power = total_energy / num_records
    cpu_power = cpu_energy / num_records
    gpu_power = gpu_energy / num_records

    print(f"{total_energy=:.2f}J")
    print(f"{num_records=}")
    print(f"{avg_power=:.3f}")
    print(f"{cpu_power=:.3f}")
    print(f"{gpu_power=:.3f}")

    with open(out_file, "a") as a:
        a.write(f"{num_records=}\n")
        a.write(f"{avg_power=:.3f}\n")
        a.write(f"{cpu_power=:.3f}\n")
        a.write(f"{gpu_power=:.3f}\n")
