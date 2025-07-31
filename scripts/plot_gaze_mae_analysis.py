import os, argparse, pathlib, matplotlib, json, math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


substitutions = {"full_real": "Full Real", "full_real_synthetic_extended": "Synth. Ext. Full Real", "full_real_synthetic_extended_large": "Large Synth. Ext. Full Real", "full_synthetic": "Full Synthetic", "mixed_real_synthetic_samesize": "Mixed Real Synthetic Same-size",
                 "synthetic_bp0": "Synthetic Non-Bright-Pupil", "synthetic_bp1": "Synthetic Bright-Pupil", "real_complement_bp0": "Real Non-Bright-Pupil Only", "real_complement_bp1": "Real Bright-Pupil Only",
                 "synthetic_closed": "Synthetic Closed", "synthetic_open": "Synthetic Open", "real_complement_open": "Real Open Only", "real_complement_closed": "Real Closed Only",
                 "synthetic_narrower": "Synthetic Narrower", "synthetic_narrow": "Synthetic Narrow", "synthetic_wide": "Synthetic Wide", "real_complement_wider": "Real Wider Only", "real_complement_narrow": "Real Narrow Only", "real_complement_wide": "Real Wide Only",
                 "synthetic_fh0": "Synthetic No Facial Hair", "synthetic_fh1": "Synthetic Facial Hair", "real_complement_fh0": "Real No Facial Hair Only", "real_complement_fh1": "Real Facial Hair Only",
                 "synthetic_gl0": "Synthetic No Glasses", "synthetic_gl1": "Synthetic Glasses", "real_complement_gl0": "Real No Glasses Only", "real_complement_gl1": "Real Glasses Only",
                 }


def main(args, pgf):
    src = pathlib.Path(args.src)
    src_files = None
    if src.name.endswith(".json"):
        src_files = [src]
    elif os.path.isdir(src):
        src_files = list(src.glob("gaze_analysis_*.json"))
    dst_dir = pathlib.Path(args.dst_dir)
    os.makedirs(dst_dir, exist_ok=True)
    
    multiplot_files = args.multiplot if args.multiplot is not None else []
    
    min_error, max_error = None, None
    if len(multiplot_files) > 0:
        src_files = [f for f in src_files if f.name in multiplot_files]
        dst_file = dst_dir / ("multiplot_" + args.mode + ("_extended" if len(multiplot_files) > 2 else "") + (".pgf" if pgf else ".png"))
        
        min_error, max_error = float("inf"), float("-inf")
        for src_file in src_files:
            sample_data = parse_sample_data(src_file)
            if args.mode == "heatmap":
                heatmap = calc_heatmap(sample_data, args.axis_bins)
            local_min_error = np.min(sample_data[:,2]) if args.mode != "heatmap" else np.min(heatmap)
            local_max_error = np.max(sample_data[:, 2]) if args.mode != "heatmap" else np.max(heatmap)
            min_error = min(local_min_error, min_error)
            max_error = max(local_max_error, max_error)
    
    w = (3 * max(1, len(multiplot_files)) + 0.5) if args.width is None else args.width
    h = 3 if args.height is None else args.height
    
    if args.multiplot is not None:
        fig = plt.figure(figsize=[w, h], dpi=100)
        fig.tight_layout()
        _, axarr = plt.subplots(1, len(multiplot_files), width_ratios=[1 for _ in range(len(multiplot_files))], height_ratios=[1], figsize=[w,h])
    
    minval = -math.pi * 0.5 if args.min_val is None else args.min_val
    maxval = math.pi * 0.5 if args.max_val is None else args.max_val
    
    for file_idx, src_file in enumerate(tqdm(src_files)):
        if args.multiplot is None:
            dst_file = dst_dir / (src_file.name.replace(".json", "").replace("gaze_analysis_", "") + "_"+args.mode + (".pgf" if pgf else ".png"))
        
        if args.multiplot is None:
            fig = plt.figure(figsize=[w, h], dpi=100)
            fig.tight_layout()
        ax = plt.gca() if args.multiplot is None else axarr[file_idx]
        
        title = src_file.name[len("gaze_analysis_vit16_gz_"):].replace(".json", "")
        if "_gh" in title:
            title = title[:title.rindex("_")]
        if title in substitutions.keys():
            title = substitutions[title]
            
        sample_data = parse_sample_data(src_file)
        heatmap = calc_heatmap(sample_data, args.axis_bins)
        
        if args.mode == "heatmap":
            ax.imshow(heatmap, cmap=args.colormap, vmin=min_error, vmax=max_error, interpolation="nearest", aspect="auto")
            
            ticks = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
            bin_ticks = np.array([radians_to_bin(x) for x in ticks])
            ax.set_xticks(bin_ticks, [str(x) for x in ticks])
            ax.set_yticks(bin_ticks, [str(x) for x in ticks])
            
            ax.set_xlim(left=radians_to_bin(minval), right=radians_to_bin(maxval))
            ax.set_ylim(bottom=radians_to_bin(minval), top=radians_to_bin(maxval))
            
            if file_idx == len(src_files)-1 or args.multiplot is None:
                cbar = plt.colorbar(plt.pcolor(heatmap, cmap=args.colormap), fraction=0.046, pad=0.04)
                if args.multiplot is not None:
                    cbar.mappable.set_clim(min_error, max_error)
                
        else:
            sc = ax.scatter(x=sample_data[:, 1], y=sample_data[:, 0], c=sample_data[:, 2], cmap=args.colormap)
            
            ax.set_xlim(left=minval, right=maxval)
            ax.set_ylim(bottom=minval, top=maxval)
            
            if file_idx == len(src_files) - 1 or args.multiplot is None:
                cbar = plt.colorbar(sc)  # , fraction=0.046, pad=0.04)
                if args.multiplot is not None:
                    cbar.mappable.set_clim(min_error, max_error)
        
        ax.set_xlabel("yaw")
        ax.set_ylabel("pitch")
        
        if title is not None:
            pady = 0
            # pady = 30 if len(x_labels) > 1 else 0
            ax.set_title(title, loc="left", pad=pady)
        
        #ax.margins(y=(0.2 if args.margin_y is None else args.margin_y))
        
        if args.multiplot is None:
            plt.savefig(dst_file, bbox_inches="tight", pad_inches=0.0)
            plt.cla()
            plt.clf()
            plt.close()
            
    if args.multiplot is not None:
        plt.savefig(dst_file, bbox_inches="tight", pad_inches=0.0)


def parse_sample_data(json_file):
    with open(json_file, "r") as f:
        sample_data_list = json.load(f)["rmse_tuples"]
        sample_data = np.empty((len(sample_data_list), 3), dtype=np.float32)
        for idx, tpl in enumerate(sample_data_list):
            sample_data[idx][0] = tpl[0][0]
            sample_data[idx][1] = tpl[0][1]
            sample_data[idx][2] = tpl[1]
    return sample_data


def calc_heatmap(sample_data, axis_bins):
    heatmap = np.zeros((axis_bins, axis_bins), dtype=np.float32)
    heatmap_sample_counter = np.zeros(heatmap.shape, dtype=np.uint32)
    for i in range(sample_data.shape[0]):
        sample = sample_data[i]
        bins = [radians_to_bin(sample[j]) for j in [1, 0]]
        
        heatmap[bins[0], bins[1]] += sample[2]
        heatmap_sample_counter[bins[0], bins[1]] += 1
    
    for x in range(heatmap.shape[0]):
        for y in range(heatmap.shape[1]):
            count = heatmap_sample_counter[x, y]
            if count > 0:
                heatmap[x, y] = heatmap[x, y] / count
                
    return heatmap


def radians_to_bin(r):
    return int(((r + math.pi * 0.5) / math.pi) * args.axis_bins)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src", type=str, required=True)
    parser.add_argument("--multiplot", type=str, nargs="+", default=None)
    parser.add_argument("-d", "--dst_dir", type=str, default=None)
    parser.add_argument("--mode", type=str, default="scatter", choices=["scatter", "heatmap"])
    parser.add_argument("--colormap", type=str, default="viridis")
    parser.add_argument("--axis_bins", type=int, default=32)
    
    parser.add_argument("--min_val", type=float, default=None)
    parser.add_argument("--max_val", type=float, default=None)
    
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--width", type=float, default=None)
    parser.add_argument("--height", type=float, default=None)
    parser.add_argument("--margin_y", type=float, default=None)
    
    args = parser.parse_args()
    
    for pgf in [False, True]:
        if pgf:
            matplotlib.use("pgf")
            matplotlib.rcParams.update({
                "pgf.texsystem": "pdflatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
                "figure.autolayout": True
            })
        else:
            matplotlib.use('TKAgg')
            matplotlib.rcParams.update({"figure.autolayout": True})
        main(args, pgf)
