import os
import subprocess
import argparse


# def train_arg_parser():

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, help="Path to dataset yaml file")
#     parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
#     parser.add_argument("--doc", type=str, required=True, help="A string for documentation purpose. Will be the name of the log folder.")
#     parser.add_argument("--sample", action="store_true", help="Whether to produce samples from the model")
#     parser.add_argument("--timesteps", type=int, default=50, help="Number of timesteps for the denoising process")
#     parser.add_argument("--eta", type=float, default=0.0)
#     parser.add_argument("--total_n_samples", type=int, default=100, help="number of examples to be generated")
#     args = parser.parse_args()
#     return args

#args = train_arg_parser()

def sample_ddim(mode, dataset, n_samples=50, timestep=50, eta=0):

    if mode != 'sample':
        if dataset == "MADISON":
            config = "/home/syurtseven/gsoc-2024/external/ddim/configs/madison.yml"
            exp    = "../models/results" 
            os.makedirs(exp, exist_ok=True)
            python_command = f"python ../external/ddim/main.py --config {config} --exp {exp} --ni --doc"
        else:
            raise(NotImplementedError)
    
    else:
        if dataset == "MADISON":
            config = "/home/syurtseven/gsoc-2024/external/ddim/configs/madison.yml"
            exp    = "../models/results" 
            doc    = "diffusion_madison"
            os.makedirs(os.path.join(exp, 'logs', doc), exist_ok=True)
            python_command =  f"python ../external/ddim/main.py --config {config} --doc {doc} --exp {exp} --sample --fid --timesteps {timestep} --eta {eta} --ni --total_n_samples {n_samples} "
        else:
            raise(NotImplementedError)

    bashcode = ''
    command = {
        f"exp":bashcode + python_command
        }


    eval = 'eval "$(conda shell.bash hook)"'

    for k, v in command.items():
        code1 = f"tmux+new-session+-d+-s+{k}"
        code2 = f"tmux+send-keys+-t+{k}+{eval}+Enter"
        code3 = f"tmux+send-keys+-t+{k}+conda activate gsoc+Enter"
        code4 = f"tmux+send-keys+-t+{k}+{v}+Enter"

        for i in [code1, code2, code3, code4]:
            res = subprocess.run(i.split('+'))
            print(res)

