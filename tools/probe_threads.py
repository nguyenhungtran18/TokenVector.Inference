# PROBE THREADS — do CPU-time (user+sys) cua tien trinh con TKVC, chia cho
# wall time => ratio = so core dang chay THAT cua pha do.
#   ratio ~1  : 1 luong (scalar)      ratio ~14: da luong 14 core
#
# Cach dung (tu goc repo):
#   printf "0\n" > tv/tests/_thr_cfg.txt     # 0=pure gemm 1=scalar 2=1 buoc
#   D:/TokenVector/Release/dist/tkvc-tv.exe tv/tests/simd_probe_ao.tkv out/ao.exe main
#   python tools/probe_threads.py out/ao.exe [reps=3]
#
# Windows: os.times() KHONG dem CPU cua con (0) -> poll psutil.Process.cpu_times()
# trong khi tien trinh chay, lay mau cuoi truoc khi no ket thuc.
import os
import subprocess
import sys
import time

import psutil


def run_once(exe):
    t0 = time.perf_counter()
    p = subprocess.Popen([exe], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         text=True)
    proc = psutil.Process(p.pid)
    cpu = 0.0
    while p.poll() is None:
        try:
            ct = proc.cpu_times()
            cpu = ct.user + ct.system
        except Exception:
            pass
        time.sleep(0.02)
    out, err = p.communicate()
    t1 = time.perf_counter()
    return t1 - t0, cpu, out.strip()


def main():
    exe = os.path.abspath(sys.argv[1])
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    best = None
    for _ in range(reps):
        wall, cpu, out = run_once(exe)
        ratio = cpu / wall if wall > 0 else 0.0
        print("THREADS wall=%.3f s  cpu=%.3f s  ratio=%.2f  | %s" % (wall, cpu, ratio, out))
        if best is None or wall < best[0]:
            best = (wall, cpu, ratio, out)
    print("THREADS BEST wall=%.3f s cpu=%.3f s ratio=%.2f | %s"
          % (best[0], best[1], best[2], best[3]))


if __name__ == "__main__":
    main()
