import json
import logging
import os
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="Failed to load image Python extension: .*"
)

import h5py
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.distributed as dist
import torchvision.transforms as T
import utils
from cctorch import CCDataset, CCIterableDataset, CCModel
from cctorch.transforms import *
from cctorch.utils import write_cc_pairs, write_tm_detects,write_ambient_noise
from sklearn.cluster import DBSCAN
from torch.utils.data import DataLoader
from tqdm import tqdm
from time import time

from cctorch import CCMapDataset

import shutil

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Cross-Correlation using Pytorch", add_help=add_help)
    parser.add_argument(
        "--mode",
        default="CC",
        type=str,
        help="mode for tasks of CC (cross-correlation), TM (template matching), and AN (ambient noise)",
    )
    parser.add_argument("--pair_list", default=None, type=str, help="pair list")
    parser.add_argument("--data_list1", default=None, type=str, help="data list 1")
    parser.add_argument("--data_list2", default=None, type=str, help="data list 1")
    parser.add_argument("--data_path1", default="./", type=str, help="data path")
    parser.add_argument("--data_path2", default="./", type=str, help="data path")
    parser.add_argument("--data_format1", default="h5", type=str, help="data type in {h5, memmap}")
    parser.add_argument("--data_format2", default="h5", type=str, help="data type in {h5, memmap}")
    parser.add_argument("--config", default=None, type=str, help="config file")
    parser.add_argument("--result_path", default="./results", type=str, help="results path")
    parser.add_argument("--dataset_type", default="iterable", type=str, help="data loader type in {map, iterable}")
    parser.add_argument(
        "--block_size1", default=1024, type=int, help="Number of sample for the 1st data pair dimension"
    )
    parser.add_argument(
        "--block_size2", default=1024, type=int, help="Number of sample for the 2nd data pair dimension"
    )
    parser.add_argument("--auto_xcorr", action="store_true", help="do auto-correlation for data list")

    ## common
    parser.add_argument("--dt", default=0.01, type=float, help="time sampling interval")
    parser.add_argument("--sampling_rate", default=100, type=float, help="sampling frequency")
    parser.add_argument("--domain", default="time", type=str, help="domain in {time, freqency, stft}")
    parser.add_argument("--maxlag", default=0.5, type=float, help="maximum time lag during cross-correlation")
    parser.add_argument("--batch_size", default=1024, type=int, help="batch size")
    parser.add_argument("--buffer_size", default=10, type=int, help="buffer size for writing to h5 file")
    parser.add_argument("--workers", default=4, type=int, help="data loading workers")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu, Default: cuda)")
    parser.add_argument(
        "--dtype", default="float32", type=str, help="data type (Use float32 or float64, Default: float32)"
    )
    parser.add_argument("--normalize", action="store_true", help="normalized cross-correlation (pearson correlation)")

    ## template matching parameters
    parser.add_argument("--shift_t", action="store_true", help="shift continuous waveform to align with template time")

    ## ambient noise parameters
    parser.add_argument("--min_channel", default=0, type=int, help="minimum channel index")
    parser.add_argument("--max_channel", default=None, type=int, help="maximum channel index")
    parser.add_argument("--delta_channel", default=1, type=int, help="channel interval")
    parser.add_argument("--left_channel", default=None, type=int, help="channel index of the left end from the source")
    parser.add_argument(
        "--right_channel", default=None, type=int, help="channel index of the right end from the source"
    )
    parser.add_argument(
        "--fixed_channels",
        nargs="+",
        default=None,
        type=int,
        help="fixed channel index, if specified, min and max are ignored",
    )
    parser.add_argument("--temporal_gradient", action="store_true", help="use temporal gradient")

    # cross-correlation parameters
    parser.add_argument("--picks_csv", default="cctorch_picks.csv", type=str, help="picks file")
    parser.add_argument("--events_csv", default="cctorch_events.csv", type=str, help="events file")
    parser.add_argument("--stations_csv", default="cctorch_stations.csv", type=str, help="stations file")
    parser.add_argument("--taper", action="store_true", help="taper two data window")
    parser.add_argument("--interp", action="store_true", help="interpolate the data window along time axs")
    parser.add_argument("--scale_factor", default=10, type=int, help="interpolation scale up factor")
    parser.add_argument(
        "--channel_shift", default=0, type=int, help="channel shift of 2nd window for cross-correlation"
    )
    parser.add_argument("--reduce_t", action="store_true", help="reduce the time axis of xcor data")
    parser.add_argument(
        "--reduce_x",
        action="store_true",
        help="reduce the station axis of xcor data: only have effect when reduce_t is true",
    )
    parser.add_argument("--reduce_c", action="store_true", help="reduce the channel axis of xcor data")
    parser.add_argument(
        "--mccc", action="store_true", help="use mccc to reduce time axis: only have effect when reduce_t is true"
    )
    parser.add_argument("--phase_type1", default="P", type=str, help="Phase type of the 1st data window")
    parser.add_argument("--phase_type2", default="S", type=str, help="Phase type of the 2nd data window")
    parser.add_argument(
        "--path_xcor_data", default="", type=str, help="path to save xcor data output: path_{channel_shift}"
    )
    parser.add_argument(
        "--path_xcor_pick", default="", type=str, help="path to save xcor pick output: path_{channel_shift}"
    )
    parser.add_argument(
        "--path_xcor_matrix", default="", type=str, help="path to save xcor matrix output: path_{channel_shift}"
    )
    parser.add_argument("--path_dasinfo", default="", type=str, help="csv file with das channel info")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    #parser.add_argument("--distributed", action="store_true", help="Enable distributed training across multiple GPUs")
    return parser

from dataclasses import dataclass
import torch

@dataclass
class CCConfig:
    mode: str
    domain: str
    dtype: torch.dtype
    device: str
    dt: float
    fs: float
    sampling_rate: float
    maxlag: float
    nlag: int
    pre_fft: bool
    auto_xcorr: bool
    spectral_whitening: bool
    max_channel: int
    min_channel: int
    delta_channel: int
    left_channel: int
    right_channel: int
    fixed_channels: int
    transform_on_file: bool
    transform_on_batch: bool
    transform_device: str
    window_size: int
    fmin: float
    fmax: float
    ftype: str
    alpha: float
    order: int
    decimate_factor: int
    nma: tuple
    reduce_t: bool
    reduce_x: bool
    reduce_c: bool
    channel_shift: bool
    mccc: bool
    use_pair_index: bool
    min_cc: float
    max_shift: dict
    max_obs: int
    min_obs: int
    shift_t: bool
    normalize: bool

    stack_channel: int

    def __init__(self, args, config=None):
        # Initialize from args
        self.mode = args.mode
        self.domain = args.domain
        self.dtype = torch.float32 if args.dtype == "float32" else torch.float64
        self.device = args.device
        self.dt = args.dt
        self.fs = args.sampling_rate
        if self.dt != 0.01:
            self.fs = 1 / self.dt
        if self.fs != 100:
            self.dt = 1 / self.fs
        self.maxlag = args.maxlag
        self.nlag = int(self.maxlag / self.dt)
        self.pre_fft = False  # if true, do fft in dataloader
        self.auto_xcorr = args.auto_xcorr
        self.spectral_whitening = True
        self.max_channel = args.max_channel
        self.min_channel = args.min_channel
        #self.delta_channel = args.delta_channel
        self.left_channel = args.left_channel
        self.right_channel = args.right_channel
        self.fixed_channels = args.fixed_channels
        # self.fixed_channels = list(np.arange(1024,10240,1024))
        # self.fixed_channels = list(np.arange(1024,10240,512))
        # self.fixed_channels = list(np.arange(1024,10240,256))
        self.transform_on_file = True
        self.transform_on_batch = False
        self.transform_device = "cuda"
        self.window_size = 64
        self.fmin = 0.1
        self.fmax = 10
        self.ftype = "bandpass"
        self.alpha = 0.05  # tukey window parameter
        self.order = 2
        self.decimate_factor = 8
        self.nlag = self.nlag // self.decimate_factor
        self.nma = (20, 0)
        self.reduce_t = args.reduce_t
        self.reduce_x = args.reduce_x
        self.reduce_c = args.reduce_c
        self.channel_shift = args.channel_shift
        self.mccc = args.mccc
        self.use_pair_index = True if args.dataset_type == "map" else False
        self.min_cc = 0.5
        self.max_shift = {"P": int(0.5 * self.fs), "S": int(0.85 * self.fs)}
        self.max_obs = 100
        self.min_obs = 8
        self.shift_t = args.shift_t
        self.normalize = args.normalize

        self.stack_channel = 5
        self.delta_channel = self.stack_channel if self.stack_channel else 1
        

        # Override with values from config if provided
        if config is not None:
            for k, v in config.items():
                setattr(self, k, v)

def identity_collate(x):
    return x



def main(args):
    logging.basicConfig(filename="cctorch.log", level=logging.INFO)
    utils.init_distributed_mode(args)
    rank = utils.get_rank() if args.distributed else 0
    world_size = utils.get_world_size() if args.distributed else 1

    if args.config is not None:
        with open(args.config, "r") as f:
            config = json.load(f)
        print(json.dumps(config, indent=4))
    else:
        config = None


    ccconfig = CCConfig(args)
    

    ## Sanity check
    if args.mode == "TM":
        pass

    if rank == 0:
        if not os.path.exists(args.result_path):
            os.makedirs(args.result_path)

    preprocess = []
    if args.mode == "CC":
        # preprocess.append(Filtering(1, 15, 100, 0.1, ccconfig.dtype, args.device))
        if args.taper:
            preprocess.append(T.Lambda(taper_time))
        if args.domain == "time":
            preprocess.append(T.Lambda(normalize))
        elif args.domain == "frequency":
            preprocess.append(T.Lambda(fft_real_normalize))
        else:
            raise ValueError(f"domain {args.domain} not supported")
    elif args.mode == "TM":
        ## TODO add preprocess for template matching
        pass
    elif args.mode == "AN":
        print('AN')

        filtering_params = {
                            "fmin": ccconfig.fmin,
                            "fmax": ccconfig.fmax,
                            "fs": ccconfig.fs,
                            "ftype": ccconfig.ftype,
                            "alpha": ccconfig.alpha,
                            "dtype": ccconfig.dtype,
                            "device": 'cuda',
                           }

        temporal_norm_params = {
                               "window_size": int(2 * ccconfig.fs // ccconfig.decimate_factor),
                               }

        temporal_gradient_params = {
                                   "fs": ccconfig.fs,
                                   }
                               
        decimation_params = {
                            "decimation": ccconfig.decimate_factor,
                            }
        

    preprocess = T.Compose(preprocess)

    postprocess = []
    if args.mode == "CC":
        ## TODO: add postprocess for cross-correlation
        postprocess.append(DetectPeaksCC(kernel=3, stride=1, topk=2))
    elif args.mode == "TM":
        postprocess.append(
            DetectPeaksTM(vmin=0.6, kernel=301, stride=1, topk=3600 // 5)
        )  # assume 100Hz and 1 hour file
    elif args.mode == "AN":
        ## TODO: add postprocess for ambient noise
        pass
    postprocess = T.Compose(postprocess)

    if args.dataset_type == "map":
        dataset = CCDataset(
            config=ccconfig,
            pair_list=args.pair_list,
            data_list1=args.data_list1,
            data_list2=args.data_list2,
            block_size1=args.block_size1,
            block_size2=args.block_size2,
            data_path1=args.data_path1,
            data_path2=args.data_path2,
            data_format1=args.data_format1,
            data_format2=args.data_format2,
            device="cpu" if args.workers > 0 else args.device,
            transforms=preprocess,
            rank=rank,
            world_size=world_size,
        )
    elif args.dataset_type == "iterable":  ## prefered
        dataset = CCIterableDataset(
            config=ccconfig,
            pair_list=args.pair_list,
            data_list1=args.data_list1,
            data_list2=args.data_list2,
            block_size1=args.block_size1,
            block_size2=args.block_size2,
            data_path1=args.data_path1,
            data_path2=args.data_path2,
            data_format1=args.data_format1,
            data_format2=args.data_format2,
            #device=args.device,
            device='cpu',
            transforms=preprocess,
            batch_size=args.batch_size,
            rank=rank,
            world_size=world_size,
        )

    else:
        raise ValueError(f"dataset_type {args.dataset_type} not supported")
    # if len(dataset) < world_size:
    #     raise ValueError(f"dataset size {len(dataset)} is smaller than world size {world_size}")

    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=args.workers if (args.dataset_type == "map") else 4,
        #num_workers=args.workers,
        sampler=sampler if args.dataset_type == "map" else None,
        #pin_memory=False,
        pin_memory=True,
        #prefetch_factor=2,
        #collate_fn=lambda x: x,
        collate_fn=identity_collate,
        #collate_fn=collate_fn,
        persistent_workers=True
    )
    ccmodel = CCModel(
        config=ccconfig,
        batch_size=args.batch_size,  ## only useful for dataset_type == map
        #to_device=False,  ## to_device is done in dataset in default
        to_device=True,  ## to_device is done in dataset in default
        device=args.device,
        transforms=postprocess,
        temporal_gradient_params=temporal_gradient_params,
        filtering_params=filtering_params,
        decimation_params=decimation_params,
        temporal_norm_params=temporal_norm_params,
    )
    
    ccmodel.to(args.device)

    if args.mode == "AN":

        result_df = []

        count = 0
        
        for i, data in enumerate(tqdm(dataloader, position=rank, desc=f"{rank}/{world_size}: computing")):

            #print(data[0])

            if args.dataset_type == 'iterable':

               #data.to(arg.device)
               t1 = time() 
               #if i!=0:
                  #print(str(i)+' dataloader time : ' + str(t1-t4))
               result = ccmodel(data) 
               t2 = time()
               print(' '+str(i)+ ' model time : ' + str(time()-t1))
               
            
            if args.dataset_type == 'map':
               result = ccmodel.forward_map(data)

        
            #print(type(result))
            t3 = time() 
            
            count += 1
            if count == 1:
               data_stack = result['xcorr']
            else:
               data = result['xcorr']
               data_stack = count / (count + 1) * data_stack[:] + data[:] / (count + 1)

            if count%60 == 0:
            # if count%1 == 0:
            
               with h5py.File(os.path.join(args.result_path, f"{ccconfig.mode}_{rank:03d}_{world_size:03d}_{i+1:02d}_stack.h5"), "w") as fp:
                    #Store each key-value pair
                    for key, value in result.items():
                        if isinstance(value, torch.Tensor):  # Convert tensors to NumPy
                            fp.create_dataset(key, data=data_stack.numpy())
                        elif isinstance(value, list):  # Convert list to NumPy
                            fp.create_dataset(key, data=np.array(value))
                        elif isinstance(value, (int, float, str)):  # Save scalars and strings
                            fp.attrs[key] = value
                        else:
                            print(f"Skipping key {key}: unsupported data type {type(value)}")
                    fp.attrs['count'] = count

               count = 0

                                        
            t4 = time()
            # print(str(i)+' write h5 time : ' + str(t4-t3))
            

if __name__ == "__main__":
    import torch.multiprocessing as mp
    #import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    args = get_args_parser().parse_args()
    # if args.distributed:
    #    dist.init_process_group(backend="nccl", init_method=args.dist_url, world_size=args.world_size)
    #    torch.cuda.set_device(rank)
    main(args)
