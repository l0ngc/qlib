import os
import subprocess
import shutil

BASE_ADDR = '/16t-2/cl/project/qlib/examples/benchmarks_dynamic/DDG-DA'
BENCH_ADDR = '/16t-2/cl/project/qlib/examples/benchmarks/'
output_dir = '/16t-2/cl/project/qlib/examples/us_day_res/DDGDA'

# models = os.listdir('/16t-2/cl/project/qlib/examples/benchmarks')
# tgt_models = ['LSTM']
tgt_models = ['LSTM']


# Assuming tgt_models and BASE_ADDR are defined earlier in the script
os.chdir(f'{BASE_ADDR}')

for model in tgt_models:
    try:
        model_dir = f'{BENCH_ADDR}/{model}'
        # conf_name = f'workflow_config_{model.lower()}_Alpha158_us.yaml'
        conf_name = f'workflow_config_{model.lower()}_Alpha360_us.yaml'

        conf_addr = f'{model_dir}/{conf_name}'

        print(conf_addr)
        print(f'python workflow_us.py --conf_path={conf_addr} run')
        # # # Running shell commands
        subprocess.run(f'rm -rf mlruns', shell=True)
        subprocess.run(f'python workflow_us.py --conf_path={conf_addr} run', shell=True)
        
        mlruns_files = os.listdir(f"./mlruns/")
        max_experiment = max(i for i in mlruns_files if i.isdigit())
        tmp_md5 =list(filter(lambda x:'.' not in x, os.listdir(f"./mlruns/{max_experiment}")))[0]
        artifacts_dir = f"./mlruns/{max_experiment}/{tmp_md5}/artifacts"

        if os.path.exists(artifacts_dir):
        # Copy artifacts to the output directory named after the model
            model_output_dir = os.path.join(output_dir, model)
            shutil.copytree(artifacts_dir, model_output_dir)
    except:
        print(f"Error in {model}")
