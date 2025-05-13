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
common_parser = parser.add_argument_group(
    title="Common arguments"
)
fpca_parser = parser.add_argument_group(
    title="Arguments specific to functional PCA"
)
spatial_ldr_parser = parser.add_argument_group(
    title="Arguments specific to constructing spatial LDRs"
)
pace_parser = parser.add_argument_group(
    title="Arguments specific to estimating functional PCs using PACE"
)
relatedness_parser = parser.add_argument_group(
    title="Arguments specific to removing genetic relatedness in LDRs"
)
gwas_parser = parser.add_argument_group(
    title="Arguments specific to doing genome-wide association analysis"
)
sumstats_parser = parser.add_argument_group(
    title="Arguments specific to organizing and preprocessing GWAS summary statistics"
)
voxelgwas_parser = parser.add_argument_group(
    title="Arguments specific to recovering voxel-level GWAS results"
)
herigc_parser = parser.add_argument_group(
    title="Arguments specific to heritability and (cross-trait) genetic correlation analysis"
)
make_mt_parser = parser.add_argument_group(
    title="Arguments specific to making a hail.MatrixTable of genotype data"
)


# module arguments
fpca_parser.add_argument(
    "--fpca", action="store_true", help="Functional PCA."
)
spatial_ldr_parser.add_argument(
    "--make-spatial-ldrs", action="store_true", help="Constructing spatial LDRs."
)
pace_parser.add_argument(
    "--pace", action="store_true", help="Estimating functional PCs using PACE."
)
relatedness_parser.add_argument(
    "--relatedness", action="store_true", help="Removing genetic relatedness in LDRs."
)
gwas_parser.add_argument(
    "--gwas", action="store_true", help="Genome-wide association analysis."
)
sumstats_parser.add_argument(
    "--sumstats",
    action="store_true",
    help="Organizing and preprocessing GWAS summary statistics.",
)
herigc_parser.add_argument(
    "--heri-gc",
    action="store_true",
    help="Heritability and (cross-trait) genetic correlation analysis.",
)
voxelgwas_parser.add_argument(
    "--voxel-gwas", action="store_true", help="Recovering voxel-level GWAS results."
)
make_mt_parser.add_argument(
    "--make-mt", action="store_true", help="Making a hail.MatrixTable of genotype data."
)


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
        "Directory to baseline covariate file. "
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
common_parser.add_argument(
    "--spatial-ldrs",
    help=(
        "Directory to spatial LDRs."
    ),
)
common_parser.add_argument(
    "--recon-ldrs",
    help=(
        "Directory to reconstructed LDRs."
    ),
)
common_parser.add_argument(
    "--ldr-sumstats",
    help=(
        "Prefix of preprocessed LDR GWAS summary statistics."
    ),
)
common_parser.add_argument(
    "--ldr-cov",
    help=(
        "Directory to variance-covariance marix of LDRs."
    ),
)
common_parser.add_argument(
    "--extract",
    help=(
        "SNP file(s). Multiple files are separated by comma. "
        "Only common SNPs appearing in all files will be extracted (logical and). "
        "Each file should be tab or space delimited, "
        "with the first column being rsID. "
        "Other columns will be ignored. "
        "Each row contains only one SNP."
    ),
)
common_parser.add_argument(
    "--exclude",
    help=(
        "SNP file(s). Multiple files are separated by comma. "
        "SNPs appearing in any files will be excluded (logical or). "
        "Each file should be tab or space delimited, "
        "with the first column being rsID. "
        "Other columns will be ignored. "
        "Each row contains only one SNP."
    ),
)
common_parser.add_argument(
    "--maf-min",
    type=float,
    help=(
        "Minimum minor allele frequency for screening variants."
    ),
)
common_parser.add_argument(
    "--maf-max",
    type=float,
    help=(
        "Maximum minor allele frequency for screening variants."
    ),
)
common_parser.add_argument(
    "--mac-min",
    type=int,
    help=(
        "Minimum minor allele count for screening variants."
    ),
)
common_parser.add_argument(
    "--mac-max",
    type=int,
    help=(
        "Maximum minor allele count for screening variants."
    ),
)
common_parser.add_argument(
    "--hwe",
    type=float,
    help=(
        "A HWE p-value threshold. "
        "Variants with a HWE p-value less than the threshold "
        "will be removed."
    ),
)
common_parser.add_argument(
    "--call-rate",
    type=float,
    help=(
        "A genotype call rate threshold, equivalent to 1 - missing rate. "
        "Variants with a call rate less than the threshold "
        "will be removed."
    ),
)
common_parser.add_argument(
    "--bfile",
    help=(
        "Prefix of PLINK bfile triplets. "
        "When estimating LD matrix and its inverse, two prefices should be provided "
        "and seperated by a comma, e.g., `prefix1,prefix2`. "
        "When doing GWAS, only one prefix is allowed."
    ),
)
common_parser.add_argument(
    "--chr-interval", "--range",
    help=(
        "A segment of chromosome, e.g. `3:1000000-2000000`, "
        "from chromosome 3 bp 1000000 to chromosome 3 bp 2000000. "
        "Cross-chromosome is not allowed. And the end position must "
        "be greater than the start position."
    ),
)
common_parser.add_argument(
    "--geno-mt",
    help=(
        "Directory to genotype MatrixTable."
    ),
)
common_parser.add_argument(
    "--grch37",
    action="store_true",
    help=(
        "Using reference genome GRCh37. Otherwise using GRCh38."
    ),
)
common_parser.add_argument(
    "--variant-type",
    help=(
        "Variant type (case insensitive), "
        "must be one of ('variant', 'snv', 'indel')."
    ),
)
common_parser.add_argument(
    "--spark-conf",
    help=(
        "Spark configuration file."
    ),
)
common_parser.add_argument(
    "--loco-preds",
    help=(
        "Leave-one-chromosome-out prediction file."
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

# arguments for herigc.py
herigc_parser.add_argument(
    "--ld-inv",
    help=(
        "Prefix of inverse LD matrix. Multiple matrices can be specified using {:}, "
        "e.g., `ld_inv_chr{1:22}_unrel`."
    ),
)
herigc_parser.add_argument(
    "--ld",
    help=(
        "Prefix of LD matrix. Multiple matrices can be specified using {:}, "
        "e.g., `ld_chr{1:22}_unrel`."
    ),
)
herigc_parser.add_argument(
    "--y2-sumstats",
    help="Prefix of preprocessed GWAS summary statistics of non-imaging traits.",
)
herigc_parser.add_argument(
    "--overlap",
    action="store_true",
    help=(
        "Flag for indicating sample overlap between LDR summary statistics "
        "and non-imaging summary statistics. Only effective if --y2-sumstats is specified."
    ),
)
herigc_parser.add_argument(
    "--heri-only",
    action="store_true",
    help=(
        "Flag for only computing voxelwise heritability "
        "and skipping voxelwise genetic correlation within images."
    ),
)

# arguments for sumstats.py
sumstats_parser.add_argument(
    "--ldr-gwas",
    help=(
        "Directory to raw LDR GWAS summary statistics files. "
        "Multiple files can be provided using {:}, e.g., `ldr_gwas{1:10}.txt`, "
        "or separated by comma, but do not mix {:} and comma together."
    ),
)
sumstats_parser.add_argument(
    "--ldr-gwas-heig",
    help=(
        "Directory to raw LDR GWAS summary statistics files produced by --gwas. "
        "Multiple files can be provided using {:}, e.g., `ldr_gwas{1:10}.parquet`, "
        "or separated by comma, but do not mix {:} and comma together. "
        "One file may contain multiple LDRs. LDRs in these files must be in order."
    ),
)
sumstats_parser.add_argument(
    "--y2-gwas", help="Directory to raw non-imaging GWAS summary statistics file."
)
sumstats_parser.add_argument("--n", type=float, help="Sample size. A positive number.")
sumstats_parser.add_argument("--n-col", help="Sample size column.")
sumstats_parser.add_argument("--chr-col", help="Chromosome column.")
sumstats_parser.add_argument("--pos-col", help="Position column.")
sumstats_parser.add_argument("--snp-col", help="SNP column.")
sumstats_parser.add_argument("--a1-col", help="A1 column. The effective allele.")
sumstats_parser.add_argument("--a2-col", help="A2 column. The non-effective allele.")
sumstats_parser.add_argument(
    "--effect-col",
    help=(
        "Genetic effect column, usually refers to beta or odds ratio, "
        "should be specified in this format `BETA,0` where "
        "BETA is the column name and 0 is the null value. "
        "For odds ratio, the null value is 1."
    ),
)
sumstats_parser.add_argument(
    "--se-col",
    help=(
        "Standard error column. For odds ratio, the standard error must be in "
        "log(odds ratio) scale."
    ),
)
sumstats_parser.add_argument("--z-col", help="Z score column.")
sumstats_parser.add_argument("--p-col", help="p-Value column.")
sumstats_parser.add_argument("--maf-col", help="Minor allele frequency column.")
sumstats_parser.add_argument("--info-col", help="INFO score column.")
sumstats_parser.add_argument(
    "--info-min", type=float, help="Minimum INFO score for screening SNPs."
)

# arguments for relatedness.py
relatedness_parser.add_argument(
    "--bsize", type=int, help="Block size of genotype blocks. Default: 5000."
)

# arguments for gwas.py
gwas_parser.add_argument(
    "--ldr-col", help="One-based LDR indices. E.g., `3,4,5,6` and `3:6`, must be consecutive"
)

# arguments for mt.py
make_mt_parser.add_argument(
    "--qc-mode", help="Genotype data QC mode, either gwas or wgs. Default: gwas."
)
make_mt_parser.add_argument(
    "--save-sparse-genotype", action="store_true", 
    help="Saving sparse genotype for rare variant analysis."
)
make_mt_parser.add_argument(
    "--vcf",
    help="Direcotory to a VCF file."
)
make_mt_parser.add_argument(
    "--skip-qc",
    action="store_true",
    help="Skipping QC genotype data."
)
make_mt_parser.add_argument(
    "--lift-over",
    help="Target reference genome, either `GRCh37` or `GRCh38`."
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
        "pace": {
            "out",
            "pace",
            "spatial_ldrs",
            "n_ldrs",
            "covar",
            "cat_covar_list",
            "keep",
            "remove",
        },
        "relatedness": {
            "relatedness",
            "out",
            "keep",
            "remove",
            "extract",
            "exclude",
            "recon_ldrs",
            "covar",
            "cat_covar_list",
            "partition",
            "maf_min",
            "maf_max",
            "mac_max",
            "mac_min",
            "variant_type",
            "hwe",
            "call_rate",
            "n_ldrs",
            "grch37",
            "geno_mt",
            "bsize",
            "spark_conf",
            "threads"
        }, 
        "gwas": {
            "out",
            "gwas",
            "keep",
            "remove",
            "extract",
            "exclude", 
            "maf_min",
            "maf_max",
            "mac_max",
            "mac_min",
            "variant_type",
            "hwe",
            "call_rate",
            "chr_interval",
            "ldr_col",
            "recon_ldrs",
            "n_ldrs",
            "grch37",
            "geno_mt",
            "covar",
            "cat_covar_list",
            "loco_preds",
            "spark_conf",
        },
        "voxel_gwas": {
            "out",
            "voxel_gwas",
            "sig_thresh",
            "voxels",
            "chr_interval",
            "extract",
            "exclude",
            "ldr_sumstats",
            "n_ldrs",
            "ldr_cov",
            "bases",
            "threads",
        },
        "sumstats": {
            "out",
            "sumstats",
            "ldr_gwas",
            "y2_gwas",
            "ldr_gwas_heig",
            "n",
            "n_col",
            "chr_col",
            "pos_col",
            "snp_col",
            "a1_col",
            "a2_col",
            "effect_col",
            "se_col",
            "z_col",
            "p_col",
            "maf_col",
            "maf_min",
            "info_col",
            "info_min",
            "threads",
        },
        "heri_gc": {
            "out",
            "heri_gc",
            "ld_inv",
            "ld",
            "y2_sumstats",
            "overlap",
            "heri_only",
            "n_ldrs",
            "ldr_sumstats",
            "bases",
            "ldr_cov",
            "extract",
            "exclude",
            "threads",
        },
        "make_mt": {
            "make_mt",
            "out",
            "keep",
            "remove",
            "extract",
            "exclude",
            "extract_locus",
            "exclude_locus",
            "bfile",
            "vcf",
            "geno_mt",
            "maf_min",
            "maf_max",
            "mac_max",
            "mac_min",
            "variant_type",
            "hwe",
            "call_rate",
            "chr_interval",
            "spark_conf",
            "qc_mode",
            "save_sparse_genotype",
            "grch37",
            "skip_qc",
            "lift_over",
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
    ds.check_existence(args.ldr_sumstats, ".snpinfo")
    ds.check_existence(args.ldr_sumstats, ".sumstats")
    ds.check_existence(args.bases)
    ds.check_existence(args.ldr_cov)
    ds.check_existence(args.covar)
    ds.check_existence(args.spatial_ldrs)
    ds.check_existence(args.spark_conf)
    ds.check_existence(args.loco_preds)
    ds.check_existence(args.geno_mt)

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
        
    if args.extract is not None:
        args.extract = split_files(args.extract)
        args.extract = ds.read_extract(args.extract)
        log.info(f"{len(args.extract)} SNP(s) in --extract (logical 'and' for multiple files).")
        
    if args.exclude is not None:
        args.exclude = split_files(args.exclude)
        args.exclude = ds.read_exclude(args.exclude)
        log.info(f"{len(args.exclude)} SNP(s) in --exclude (logical 'or' for multiple files).")

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

    if args.maf_min is not None:
        if args.maf_min >= 0.5 or args.maf_min < 0:
            raise ValueError("--maf-min must be >= 0 and < 0.5")
        if args.maf_max is None:
            args.maf_max = 0.5
    if args.maf_max is not None:
        if args.maf_max > 0.5 or args.maf_max <= 0:
            raise ValueError("--maf-max must be > 0 and <= 0.5")
        if args.maf_min is None:
            args.maf_min = 0
    if args.mac_min is not None and args.mac_min < 0:
        raise ValueError("--mac-min must be >= 0")
    if args.mac_max is not None:
        if args.mac_max <= 0:
            raise ValueError("--mac-max must be > 0")
        if args.mac_min is None:
            args.maf_min = 0
    if args.hwe is not None and args.hwe <= 0:
        raise ValueError("--hwe must be greater than 0")
    if args.call_rate is not None and args.call_rate <= 0:
        raise ValueError("--call-rate must be greater than 0")
    
    if args.variant_type is not None:
        args.variant_type = args.variant_type.lower()
        if args.variant_type not in {"snv", "variant", "indel"}:
            raise ValueError(
                "--variant-type must be one of ('variant', 'snv', 'indel')"
            )
    

def main(args, log):
    dirname = os.path.dirname(args.out)
    if dirname != "" and not os.path.exists(dirname):
        raise ValueError(f"{os.path.dirname(args.out)} does not exist")
    if (
        args.fpca
        + args.make_spatial_ldrs
        + args.pace
        + args.relatedness
        + args.gwas
        + args.sumstats
        + args.voxel_gwas
        + args.heri_gc
        + args.make_mt
        != 1
    ):
        raise ValueError(
            (
                "must raise one and only one of following module flags: "
                "--fpca, --make-spatial-ldrs, --pace, --relatedness, --gwas, "
                "--sumstats, --voxel-gwas, --heri-gc, --make-mt"
            )
        )

    if args.fpca:
        check_accepted_args("fpca", args, log)
        import script.fpca as module
    if args.make_spatial_ldrs:
        check_accepted_args("make_spatial_ldrs", args, log)
        import script.spatial_ldr as module
    if args.pace:
        check_accepted_args("pace", args, log)
        import script.pace as module
    if args.relatedness:
        check_accepted_args("relatedness", args, log)
        import script.relatedness as module
    if args.gwas:
        check_accepted_args("gwas", args, log)
        import script.gwas as module
    if args.sumstats:
        check_accepted_args("sumstats", args, log)
        import script.sumstats as module
    if args.voxel_gwas:
        check_accepted_args("voxel_gwas", args, log)
        import script.voxel_gwas as module
    if args.heri_gc:
        check_accepted_args("heri_gc", args, log)
        import script.herigc as module
    if args.make_mt:
        check_accepted_args("make_mt", args, log)
        import script.mt as module

    process_args(args, log)
    module.run(args, log)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.out is None:
        args.out = "main"

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