from argparse import ArgumentParser
import sys
import os
import subprocess as sp
from tqdm import tqdm
import fast


def main(args):
    # get all relevant files in directory
    paths = []
    for root, dirs, files in os.walk(args.input, topdown=False):
        for name in files:
            # only process files with this file extension
            if any(name.endswith(s) for s in [".vsi", ".tif", ".tiff", ".svs"]):
                # skip Overview images (only process full WSIs)
                if "Overview" in name:
                    continue
                paths.append(os.path.join(root, name))

    for path in tqdm(paths, "WSI"):
        try:
            id_ = path.split("/")[-1].split(".")[0]
            head_ = path.split(args.input)[-1].split(id_)[0]

            curr_output_path = args.output + "/" + head_

            # skip if path already exists
            if os.path.exists(curr_output_path + id_ + ".h5"):
                continue

            # create output folder if not exist
            os.makedirs(curr_output_path, exist_ok=True)

            pipeline = fast.Pipeline(args.fpl, {"wsi": path, "outputFilename": curr_output_path + id_})
            pipeline.parse()
            pipeline.getProcessObject("pwExporter").run()
        except Exception as e:
            print(str(e) + " | " + "Processing failed on: " + path)
            print("Processing failed on:", path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input', '-i', type=str, required=True,
                        help="full path to which dataset to run FPL on.")
    parser.add_argument('--output', '-o', type=str, required=True,
                        help="path to output directory for where to store predictions.")
    parser.add_argument('--fpl', '-p', type=str, required=True,
                        help="number of clusters to use for clustering model.")
    parser.add_argument('--verbose', '-v', type=int, default=0,
                        help='set pyFAST verbosity.')

    args = parser.parse_args(sys.argv[1:])
    print(args)

    if args.verbose == 0:
        fast.Reporter.setGlobalReportMethod(fast.Reporter.NONE)
    elif args.verbose == 1:
        fast.Reporter.setGlobalReportMethod(fast.Reporter.COUT)
    else:
        raise ValueError

    main(args)
