import subprocess

def run_command(command, description):
    """
    Run a shell command and handle errors.
    
    Args:
        command (list): Command and arguments to run
        description (str): Description of the command for logging
        
    Returns:
        bool: True if command succeeded, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(command, check=True)
        print("SUCCESS: Command completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        return False

