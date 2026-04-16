#!/usr/bin/python3

import os
import sys
import shlex
import select
import logging
import subprocess


# Configure the basic logging settings
logging.basicConfig(
    level=logging.DEBUG,  # Set the minimum level to log
    datefmt="%Y-%m-%d %H:%M:%S",
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],  # Log to the console (stdout/stderr)
)
log = logging.getLogger(__name__)


def runCmdGetSdtoutSdterr(
    *cmd,
    cwd=None,
    env=None,
    stdin=None,
    timeout=1200,  # default console timeout in seconds
    quiet=False,
    **kwargs,
):
    """Executes Cmd on the current node:
    *cmd[str-varargs]: of cmd and its arguments
    cwd[str]: current working dirpath from where cmd should run
    env[dict]: extra environment variable to be passed to the cmd
    stdin[str]: input to the cmd via its stdin
    timeout[int]: min time to wait before killing the process when no activity observed
    quiet[bool]: to skip all logs while cmd execution
    """
    # console prints to log all the running cmds for easy repro of test steps
    envStr = ""
    if env:  # for printing the extra envs
        for key, value in env.items():
            envStr += f'{key}="{value}" '
    log.info(f"++Exec [{cwd}]$ {envStr}{shlex.join(cmd)}")
    # handling extra env variables along with session envs
    if env:
        env = {k: str(v) for k, v in env.items()}
        env.update(os.environ)
    # launch process with enabled stream redirections
    process = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdin=subprocess.PIPE if stdin else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        close_fds=True,
        **kwargs,
    )
    # if enabled, the stdin input will write to the subprocess stdin
    if stdin:
        with process.stdin:
            data = stdin if isinstance(stdin, bytes) else stdin.encode()
            process.stdin.write(data)
    # make process stdout / stderr as non-blocking to make unblocked reads
    os.set_blocking(process.stdout.fileno(), False)
    os.set_blocking(process.stderr.fileno(), False)

    # live collection of process stdout / stderr streams
    def _readStream(fd):
        chunk = fd.read().strip()
        if not quiet:
            sys.stdout.write(chunk.decode())
            sys.stdout.flush()
        return chunk

    ret, stdout, stderr = None, b"", b""
    chunk = None
    while chunk != b"":  # loop reading till end of stream
        # select helps in efficient wait on resource events
        readFds = select.select([process.stdout, process.stderr], [], [], timeout)[0]
        if not readFds:
            msg = f"Reached Timeout of {timeout} sec, Exiting..."
            log.warning(msg)
            stdout += msg.encode()  # appending timeout msg to stdout for reporting
            process.kill()
            break
        # live reading of stdout
        if process.stdout in readFds:
            stdout += (chunk := _readStream(process.stdout))
        # live reading of stderr
        if process.stderr in readFds:
            stderr += (chunk := _readStream(process.stderr))
    # handling return value
    ret = process.wait()
    status = "success" if ret == 0 else "failed"
    log.info(f"[{shlex.join(cmd)}] {status} return code: {ret}")

    # cleaning ansi escape sequences
    def _cleanAnsiEscapes(stream):
        return re.sub(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])", "", stream.decode())

    return ret, _cleanAnsiEscapes(stdout), _cleanAnsiEscapes(stderr)


def runCmdGetOutput(*args, **kwargs):
    """Executes Cmd on the current node:
    return retval, stdout and stderr combined as output
    """
    ret, stdout, stderr = runCmdGetSdtoutSdterr(*args, **kwargs)
    return ret, stdout + stderr


def runCmd(*args, **kwargs):
    """Executes Cmd on the current node:
    return retval
    """
    ret, stdout, stderr = runCmdGetSdtoutSdterr(*args, **kwargs)
    return ret


def normBps(bps, multiplier):
    """Normalizes bytes per second in its integer form"""
    mulMap = {"K": 10, "M": 20, "G": 30, "T": 40}
    return int(float(bps) * (2 ** mulMap[multiplier]))


def normIps(bps, multiplier):
    """Normalizes items per second in its integer form"""
    mulMap = {"K": 3, "M": 4, "G": 5, "T": 6}
    return int(float(bps) * (10 ** mulMap[multiplier]))
