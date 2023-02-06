# Long-term fuzzing could be aborted by various reasons, e.g., OOM, crash, etc.
# To obtain longer experimental results, we need to merge multiple runs.

import os
import shutil

from tqdm import tqdm

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=str, required=True, help="Folder to all the tests."
    )
    parser.add_argument(
        "--to-merge", nargs="+", required=True, help="Folders to merge."
    )
    parser.add_argument(
        "--limit-second", type=int, default=4 * 3600, help="Time limit for total tests."
    )
    parser.add_argument("--mv", action="store_true", help="Move instead of copy.")

    args = parser.parse_args()

    print("@ Merging", args.to_merge, " --into--> ", args.root)

    assert os.path.isdir(args.root)
    for to_merge in args.to_merge:
        assert os.path.isdir(to_merge)

    time_frontier = 0

    # subfolders in args.root
    for dir in os.listdir(args.root):
        if dir != "coverage":
            time_frontier = max(time_frontier, float(dir))

    # subfolders in args.to_merge
    for to_merge in args.to_merge:
        max_time_diff = 0
        for dir in tqdm(os.listdir(to_merge)):
            if dir != "coverage":
                time_diff = float(dir)
                new_time = time_frontier + time_diff
                if new_time > args.limit_second:
                    continue
                max_time_diff = max(max_time_diff, time_diff)
                new_folder_name = f"{new_time :.3f}"

                old_dir = os.path.join(to_merge, dir)
                new_dir = os.path.join(args.root, new_folder_name)

                if args.mv:
                    shutil.move(old_dir, new_dir)
                else:
                    shutil.copytree(old_dir, new_dir)
        time_frontier += max_time_diff

    print(f"Last time time_frontier = {time_frontier}")
