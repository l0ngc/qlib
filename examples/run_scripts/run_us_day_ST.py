import os
import subprocess
import shutil

BASE_ADDR = '/16t-2/cl/project/qlib/examples/benchmarks'
OUTPUT_DIR = '/16t-2/cl/project/qlib/examples/us_day_res/ST'
# TGT_MODELS = ['Linear', 'LightGBM', 'Transformer', 'LSTM', 'XGBoost', 'MLP']
TGT_MODELS = ['Transformer']

# Create the output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_shell_command(command):
    """Run a shell command and handle errors."""
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}\n{e}")

def get_largest_run_directory(mlruns_dir):
    """Get the directory of the largest run."""
    max_size = 0
    largest_run_dir = None
    
    for root, dirs, _ in os.walk(mlruns_dir):
        for d in dirs:
            run_dir = os.path.join(root, d)
            size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(run_dir) for f in filenames)
            if size > max_size:
                max_size = size
                largest_run_dir = run_dir
    return largest_run_dir

def copy_artifacts(source, destination):
    """Copy artifacts to the destination directory."""
    if os.path.exists(destination):
        shutil.rmtree(destination)  # 删除目标目录及其所有内容    
    if os.path.exists(source):
        shutil.copytree(source, destination)
        print(f'Artifacts from {source} copied to {destination}')
    else:
        print(f'Artifacts directory {source} does not exist')

for model in TGT_MODELS:
    model_dir = os.path.join(BASE_ADDR, model)
    print(model_dir)
    os.chdir(model_dir)
    
    conf_name = f'workflow_config_{model.lower()}_Alpha158_us.yaml'
    
    # Run shell commands
    run_shell_command('rm -rf mlruns')
    run_shell_command(f'qrun {conf_name}')
    
    mlruns_dir = os.path.join(model_dir, 'mlruns')
    
    if os.path.exists(mlruns_dir):
        largest_run_dir = get_largest_run_directory(mlruns_dir)
        
        if largest_run_dir:
            tmp_md5 =list(filter(lambda x:'.' not in x, os.listdir(largest_run_dir)))[0]
            artifacts_dir = os.path.join(largest_run_dir, tmp_md5, 'artifacts')
            model_output_dir = os.path.join(OUTPUT_DIR, model)
            # print(f'Copying artifacts from {artifacts_dir} to {model_output_dir}')
            copy_artifacts(artifacts_dir, model_output_dir)
        else:
            print(f'No runs found in {mlruns_dir} for model {model}')
    else:
        print(f'{mlruns_dir} does not exist for model {model}')
