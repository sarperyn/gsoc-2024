import subprocess
########################################################################
########################################################################
######## SIMULTANEOUS EXPERIMENT script
########################################################################
########################################################################

exp_h = [
    "python train_unet.py --device cuda:0 --exp_id test_size/1 --bs 4 --epoch 51 --test_size 0.10",
    "python train_unet.py --device cuda:0 --exp_id test_size/2 --bs 4 --epoch 51 --test_size 0.20",
    "python train_unet.py --device cuda:0 --exp_id test_size/3 --bs 4 --epoch 51 --test_size 0.30",
    "python train_unet.py --device cuda:0 --exp_id test_size/4 --bs 4 --epoch 51 --test_size 0.40",
    "python train_unet.py --device cuda:1 --exp_id test_size/5 --bs 4 --epoch 51 --test_size 0.50",
    "python train_unet.py --device cuda:1 --exp_id test_size/6 --bs 4 --epoch 51 --test_size 0.60",
    "python train_unet.py --device cuda:1 --exp_id test_size/7 --bs 4 --epoch 51 --test_size 0.70",
    "python train_unet.py --device cuda:1 --exp_id test_size/8 --bs 4 --epoch 51 --test_size 0.80",
    #"python train_unet.py --device cuda:1 --exp_id test_size/9 --bs 4 --epoch 51 --test_size 0.90",

    # "python train_unet.py --device cuda:0 --exp_id overfit/0 --bs 1 --epoch 51 --test_size 0.30",
    # "python train_unet.py --device cuda:0 --exp_id overfit/1 --bs 1 --epoch 51 --test_size 0.30",
    # "python train_unet.py --device cuda:0 --exp_id overfit/2 --bs 1 --epoch 51 --test_size 0.30",
    # "python train_unet.py --device cuda:0 --exp_id overfit/3 --bs 1 --epoch 51 --test_size 0.30",
    # "python train_unet.py --device cuda:0 --exp_id overfit/4 --bs 1 --epoch 51 --test_size 0.30",
    # "python train_unet.py --device cuda:0 --exp_id overfit/5 --bs 1 --epoch 51 --test_size 0.30",
    # "python train_unet.py --device cuda:0 --exp_id overfit/6 --bs 1 --epoch 51 --test_size 0.30",
    # "python train_unet.py --device cuda:0 --exp_id overfit/7 --bs 1 --epoch 51 --test_size 0.30",
    # "python train_unet.py --device cuda:0 --exp_id overfit/8 --bs 1 --epoch 51 --test_size 0.30",
]

def main():
    
    bashcode = ''
    python_commands = exp_h

    commands = {
        f"exp-{i}":bashcode + el for i,el in enumerate(python_commands)
    }
    
    eval = 'eval "$(conda shell.bash hook)"'
    for k,v in commands.items():
        
        code1 = f"tmux+new-session+-d+-s+{k}"
        code2 = f"tmux+send-keys+-t+{k}+{eval}+Enter"
        code3 = f"tmux+send-keys+-t+{k}+conda activate gsoc+Enter"
        code4 = f"tmux+send-keys+-t+{k}+{v}+Enter"

        for i in [code1, code2, code3, code4]:
            res = subprocess.run(i.split('+'))
            print(res)


if __name__ == '__main__':
    main()