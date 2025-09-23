# import logger
import logging
import subprocess
import time
import traceback
from typing import List

logger = logging.getLogger(__name__)


def handle_subprocess(subprocess_args: List[str], check_output: bool = True):
    output = 1
    if check_output:
        try:
            # Start the subprocess
            process = subprocess.Popen(
                subprocess_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )

            output = ""

            # Read and print stdout in real-time
            for line in iter(process.stdout.readline, ""):
                print(line, end="", flush=True)  # Print each line as it's received
                output += line  # Append each line to the output variable

            # Wait for the process to complete
            process.stdout.close()
            process.wait()

            # # Check for errors in stderr
            # if process.returncode != 0:
            #     # Read stderr and log it
            #     error_output = process.stderr.read()
            #     print(f"Error occurred:\n{error_output}", flush=True)
            #     raise subprocess.CalledProcessError(process.returncode, subprocess_args, output=error_output)

        except subprocess.CalledProcessError as e:
            # Log the error details with a stack trace
            print("Subprocess failed with an error!")
            print(f"Command: {e.cmd}")
            print(f"Return code: {e.returncode}")
            print(f"Output: {e.output}")
            print("Stack trace:")
            traceback.print_exc()

        except Exception:
            # Catch other exceptions and log stack trace
            print("An unexpected error occurred!")
            print("Stack trace:")
            traceback.print_exc()
    else:
        try:
            process = subprocess.Popen(subprocess_args)
            while process.poll() is None:
                time.sleep(1)
            logger.info("Generation process has finished.")
            output = 0
        except KeyboardInterrupt:
            process.terminate()
    return output
