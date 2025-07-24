import os
import subprocess
import sys

def run_command(command):
    try:
        subprocess.check_call(command, shell=True)
        print(f"Successfully ran: {command}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to run command: {command}\nError: {e}")
        sys.exit(1)