import os
import time
import argparse
import traceback
import numpy as np
import script.dataset as ds
from script.utils import GetLogger, sec_to_str


parser = argparse.ArgumentParser(
    description=f"\n longtitudinal heritability \n"
)
common_parser = parser.add_argument_group(title="Common arguments")
fpca_parser = parser.add_argument_group(title="Arguments specific to functional PCA")
spatial_ldr_parser = parser.add_argument_group(title="Arguments specific to constructing spatial LDRs")

# module arguments
fpca_parser.add_argument("--fpca", action="store_true", help="Functional PCA.")
spatial_ldr_parser.add_argument("--make-spatial-ldr", action="store_true", help="Constructing spatial LDRs.")

# common arguments
common_parser.add_argument("--out", help="Prefix of output.")
common_parser.add_argument(
    "--image",
    help=(
        "Directory to processed raw images in HDF5 format."
    ),
)
common_parser.add_argument(
    "--n-ldrs",
    type=int,
    help=(
        "Number of LDRs."
    ),
)
common_parser.add_argument(
    "--keep",
    help=(
        "Subject ID file(s). Multiple files are separated by comma. "
        "Only common subjects appearing in all files will be kept (logical and). "
        "Each file should be tab or space delimited, "
        "with the first column being FID and the second column being IID. "
        "Other columns will be ignored. "
        "Each row contains only one subject."
    ),
)
common_parser.add_argument(
    "--remove",
    help=(
        "Subject ID file(s). Multiple files are separated by comma. "
        "Subjects appearing in any files will be removed (logical or). "
        "Each file should be tab or space delimited, "
        "with the first column being FID and the second column being IID. "
        "Other columns will be ignored. "
        "Each row contains only one subject. "
        "If a subject appears in both --keep and --remove, --remove takes precedence."
    ),
)
common_parser.add_argument(
    "--time",
    help=(
        "Time points to keep. Can be a list of time points separated by comma."
    ),
)
common_parser.add_argument(
    "--voxels", "--voxel",
    help=(
        "One-based index of voxel or a file containing voxels."
    ),
)
common_parser.add_argument(
    "--threads",
    type=int,
    help=(
        "number of threads."
    ),
)
common_parser.add_argument(
    "--covar",
    help=(
        "Directory to covariate file. "
        "The file should be tab or space delimited, with each row only one subject."
    ),
)
common_parser.add_argument(
    "--cat-covar-list",
    help=(
        "List of categorical covariates to include in the analysis. "
        "Multiple covariates are separated by comma."
    ),
)
common_parser.add_argument(
    "--time-varying-covar-list",
    help=(
        "List of time varying covariates to include in the analysis. "
        "Multiple covariates are separated by comma."
    ),
)
common_parser.add_argument(
    "--bases",
    help=(
        "Directory to functional bases."
    ),
)

# arguments for fpca.py
fpca_parser.add_argument(
    "--all-pc",
    action="store_true",
    help=(
        "Flag for generating all principal components which is min(n_subs, n_voxels), "
        "which may take longer time and very memory consuming."
    ),
)
fpca_parser.add_argument(
    "--bw-opt",
    type=float,
    help=(
        "The bandwidth you want to use in kernel smoothing. "
        "HEIG will skip searching the optimal bandwidth. "
        "For images of any dimension, just specify one number, e.g, 0.5 "
        "for 3D images."
    ),
)
fpca_parser.add_argument(
    "--skip-smoothing",
    action='store_true',
    help=(
        "Skipping kernel smoothing. "
    ),
)


def check_accepted_args(module, args, log):
    """
    Checking if the provided arguments are accepted by the module

    """
    accepted_args = {
        "fpca": {
            "out",
            "fpca",
            "image",
            "time",
            "voxels",
            "all_pc",
            "n_ldrs",
            "keep",
            "remove",
            "bw_opt",
            "skip_smoothing",
        },
        "make_spatial_ldr": {
            "out",
            "make_spatial_ldr",
            "image",
            "time",
            "voxels",
            "bases",
            "n_ldrs",
            "covar",
            "cat_covar_list",
            "keep",
            "remove",
            "threads",
        },
    }

    ignored_args = []
    for k, v in vars(args).items():
        if v is None or not v:
            continue
        elif k not in accepted_args[module]:
            ignored_args.append(k)
            setattr(args, k, None)

    if len(ignored_args) > 0:
        ignored_args = [f"--{arg.replace('_', '-')}" for arg in ignored_args]
        ignored_args_str = ", ".join(ignored_args)
        log.info(
            f"WARNING: {ignored_args_str} ignored by --{module.replace('_', '-')}."
        )

def split_files(arg):
    files = arg.split(",")
    for file in files:
        ds.check_existence(file)
    return files


def process_args(args, log):
    """
    Checking file existence and processing arguments

    """
    ds.check_existence(args.image)

    if args.n_ldrs is not None and args.n_ldrs <= 0:
        raise ValueError("--n-ldrs must be greater than 0")

    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads must be greater than 0")
    else:
        args.threads = 1
    log.info(f"Using {args.threads} thread(s) in analysis.")

    if args.keep is not None:
        args.keep = split_files(args.keep)
        args.keep = ds.read_keep(args.keep)
        log.info(f"{len(args.keep)} subject(s) in --keep (logical 'and' for multiple files).")

    if args.remove is not None:
        args.remove = split_files(args.remove)
        args.remove = ds.read_remove(args.remove)
        log.info(f"{len(args.remove)} subject(s) in --remove (logical 'or' for multiple files).")

    if args.voxels is not None:
        try:
            args.voxels = np.array(
                [int(voxel) - 1 for voxel in ds.parse_input(args.voxels)]
            )
        except ValueError:
            ds.check_existence(args.voxels)
            args.voxels = ds.read_voxel(args.voxels)
        if np.min(args.voxels) <= -1:
            raise ValueError("voxel index must be one-based")
        log.info(f"{len(args.voxels)} voxel(s) in --voxels.")
        
    if args.time is not None:
        try:
            args.time = np.array([float(time) for time in args.time.split(",")])
        except ValueError:
            raise ValueError("--time must be a list of time points separated by comma")
        log.info(f"{len(args.time)} time point(s) in --time.")
    

def main(args, log):
    dirname = os.path.dirname(args.out)
    if dirname != "" and not os.path.exists(dirname):
        raise ValueError(f"{os.path.dirname(args.out)} does not exist")
    if (
        args.fpca
        + args.make_spatial_ldr
        != 1
    ):
        raise ValueError(
            (
                "must raise one and only one of following module flags: "
                "--fpca, --make-spatial-ldr"
            )
        )

    if args.fpca:
        check_accepted_args("fpca", args, log)
        import script.fpca as module
    if args.make_spatial_ldr:
        check_accepted_args("make_spatial_ldr", args, log)
        import script.spatial_ldr as module

    process_args(args, log)
    module.run(args, log)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.out is None:
        args.out = "heig"

    logpath = os.path.join(f"{args.out}.log")
    log = GetLogger(logpath)

    # log.info(MASTHEAD)
    start_time = time.time()
    try:
        defaults = vars(parser.parse_args(""))
        opts = vars(args)
        non_defaults = [x for x in opts.keys() if opts[x] != defaults[x]]
        header = "main.py \\\n"
        options = [
            "--" + x.replace("_", "-") + " " + str(opts[x]) + " \\"
            for x in non_defaults
        ]
        header += "\n".join(options).replace(" True", "").replace(" False", "")
        header = header + "\n"
        log.info(header)
        main(args, log)
    except Exception:
        log.info(traceback.format_exc())
        raise
    finally:
        log.info(f"\nAnalysis finished at {time.ctime()}")
        time_elapsed = round(time.time() - start_time, 2)
        log.info(f"Total time elapsed: {sec_to_str(time_elapsed)}")