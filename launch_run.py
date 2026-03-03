import subprocess
from pathlib import Path
import os
import numpy as np

def launch_one(fixed_channels,gpu_id):
    """
    fixed_channels: list[int]
    """

    # sort to guarantee correct first/last
    fixed_channels = sorted(fixed_channels)

    first_ch = fixed_channels[0]
    last_ch  = fixed_channels[-1]

    out_name = f"2022360_2023005_ch_{first_ch}_{last_ch}"
    out_dir = Path("./cc_results") / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "-u", "run_AN.py",
        "--result_path", str(out_dir),
        "--data_list1", "tests/data_am1_2022360_2023005.txt",
        "--data_path1", "./temp_data_link",
        "--dt", "0.005",
        "--mode", "AN",
        "--dataset_type", "iterable",
        "--domain", "stft",
        "--min_channel", "5",
        "--max_channel", "10245",
        "--device", "cuda",
        "--block_size1", str(len(fixed_channels)),
        "--block_size2", "4096",
        "--maxlag", "16",
        "--fixed_channels", *map(str, fixed_channels),

    ]

    # "--fmin", "0.01",
    # "--fmax", "10",
    # "--ftype", "bandpass",
    # "--alpha", "0.05", 
    # "--order", "2",   
    # "--decimate_factor", "8",

    log_path = out_dir / "run.log"

    with open(log_path, "w") as f:
        subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
            start_new_session=True,
        )
# 01234 56789
if __name__ == "__main__":

    a = 2
    b = a + 5*16
    c = b + 5*16
    
    runs = [
        ([int(x) for x in np.arange(a, a+5*16,5)], 0),
        ([int(x) for x in np.arange(b, b+5*16,5)], 1),
        ([int(x) for x in np.arange(c, c+5*16,5)], 2),
    ]

    # runs = [
    #     ([7000], 2),
    # ]

    for fixed_channels, gpu_id in runs:
        launch_one(fixed_channels, gpu_id)