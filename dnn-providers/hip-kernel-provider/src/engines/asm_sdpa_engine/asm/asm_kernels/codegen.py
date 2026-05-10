# SPDX-License-Identifier: MIT
# Copyright (C) 2018-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import glob
import os
import re
import sys
import csv
from collections import defaultdict

# Cells must contain only safe ASCII identifier characters — no quotes or
# backslashes, which would break the double-quoted C++ string literals that
# codegen emits.  Digits-only cells (emitted as int literals) are exempt.
_CELL_RE = re.compile(r"^[A-Za-z0-9_./-]+$")


def _validate_csv_cell(value: str, filepath: str, row_num: int, col: str) -> None:
    if str(value).replace(".", "", 1).lstrip("-").isdigit():
        return
    if not _CELL_RE.match(value):
        print(
            f"ERROR: {filepath}:{row_num}: column '{col}' contains an illegal "
            f"character (quotes or backslashes are not allowed): {value!r}"
        )
        sys.exit(1)


this_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.basename(this_dir)
archs = [el for el in os.environ["AITER_GPU_ARCHS"].split(";")]
archs_supported = [
    os.path.basename(os.path.normpath(path)) for path in glob.glob(f"{this_dir}/*/")
]


content = """// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
// NOLINTBEGIN(readability-identifier-naming)
#pragma once
#include <array>

"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate",
        description="gen API for asm Bf16_gemm kernel",
    )
    parser.add_argument(
        "-m",
        "--module",
        required=True,
        help="""module of ASM kernel,
            e.g.: -m bf16gemm
        """,
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="aiter/jit/build",
        required=False,
        help="write all the blobs into a directory",
    )
    args = parser.parse_args()
    cfgs = []

    csv_groups = defaultdict(list)
    for arch in archs_supported:
        for el in glob.glob(
            f"{this_dir}/{arch}/{args.module}/**/*.csv", recursive=True
        ):
            cfgname = os.path.basename(el).split(".")[0]
            csv_groups[cfgname].append({"file_path": el, "arch": arch})

    ## deal with same name csv
    cfgs = []
    have_get_header = False
    for cfgname, file_info_list in csv_groups.items():
        dfs = []
        headers_list = []
        for file_info in file_info_list:
            single_file = file_info["file_path"]
            arch = file_info["arch"]
            fieldnames = []
            df = []
            with open(single_file) as file:
                dictreader = csv.DictReader(file)
                fieldnames = dictreader.fieldnames
                df = list(dictreader)
            for row_num, row in enumerate(df, start=2):
                for col, value in row.items():
                    _validate_csv_cell(value, single_file, row_num, col)
            # check headers
            required_columns = {"knl_name", "co_name"}
            if not headers_list:
                headers_list = fieldnames
                if not required_columns.issubset(headers_list):
                    missing = required_columns - set(headers_list)
                    print(
                        f"ERROR: Invalid assembly CSV format -- {single_file}. Missing required columns: {', '.join(missing)}"
                    )
                    sys.exit(1)
            elif headers_list != fieldnames:
                print(f"ERROR: CSV headers don't match in -- {cfgname}.")
                sys.exit(1)

            dicts = [d | {"arch": arch} for d in df]
            dfs += dicts
        if dfs:
            relpath = os.path.relpath(
                os.path.dirname(single_file), f"{this_dir}/{arch}"
            )
            if not have_get_header:
                required_columns = {"knl_name", "co_name", "arch"}
                other_columns = [
                    col for col in headers_list if col not in required_columns
                ]
                other_columns_comma = ", ".join(other_columns)
                sample_row = dfs[0]
                other_columns_cpp_def = "\n".join(
                    [
                        f"    {'int' if str(sample_row[col]).replace('.', '', 1).lstrip('-').isdigit() else 'std::string'} {col};"
                        for col in other_columns
                    ]
                )
                content += f"""
#define ADD_CFG({other_columns_comma}, arch, path, knl_name, co_name)         \\
    {{ knl_name, std::string(arch) + "/" + std::string(path) + co_name, arch, {other_columns_comma} }}

struct {args.module}Config
{{
    std::string knl_name;
    std::string co_name;
    std::string arch;
{other_columns_cpp_def}
}};

"""
                have_get_header = True
            cfg = [
                "ADD_CFG("
                + ", ".join(
                    (
                        f"{int(row[col]):>4}"
                        if str(row[col]).replace(".", "", 1).isdigit()
                        else f'"{row[col]}"'
                    )
                    for col in other_columns
                )
                + f', "{row["arch"]}", "{relpath}/", "{row["knl_name"]}", "{row["co_name"]}"),'
                for row in dfs
                if row["arch"] in archs
            ]
            cfg_count = len(cfg)
            cfg_txt = "\n    ".join(cfg) + "\n"

            txt = f"""static constexpr std::size_t k_{cfgname}_count = {cfg_count};
static std::array<{args.module}Config, k_{cfgname}_count> cfg_{cfgname} = {{{{
    {cfg_txt}}}}};"""
            cfgs.append(txt)

    content += "\n".join(cfgs) + "\n"
    content += "\n// NOLINTEND(readability-identifier-naming)\n"

    with open(f"{args.output_dir}/asm_{args.module}_configs.hpp", "w") as f:
        f.write(content)
