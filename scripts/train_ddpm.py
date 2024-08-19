import os
import subprocess
import argparse


def train_arg_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to dataset yaml file")
    parser.add_argument("--exp", type=str, default="exp", help="Path for saving running related data.")
    parser.add_argument("--doc", type=str, required=True, help="A string for documentation purpose. Will be the name of the log folder.")
    parser.add_argument("--sample", action="store_true", help="Whether to produce samples from the model")
    parser.add_argument("--timesteps", type=int, default=1000, help="number of steps involved")
    parser.add_argument("--eta", type=float, default=0.0, help="eta used to control the variances of sigma")
    parser.add_argument("--fid", action="store_true")
    parser.add_argument("--ni", action="store_true", help="No interaction. Suitable for Slurm Job launcher")
    args = parser.parse_args()
    return args

args = train_arg_parser()
train_command = [
    f"python ../external/ddim/main.py --config {args.config} --exp {args.exp} --ni"
    ]

sample_command = [
    f"python ../external/ddim/main.py --config {args.path_yml} --exp {args.exp} --sample --fid --timesteps {args.step} --eta {args.eta} --ni"
]

def main():
    bashcode = ''
    python_command = sample_command if args.sample else train_command

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


if __name__ == '__main__':
    main()