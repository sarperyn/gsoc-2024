import subprocess
########################################################################
########################################################################
######## SIMULTANEOUS EXPERIMENT script
########################################################################
########################################################################

exp_h = [
    "python train_unet.py --device cuda:1 --exp_id levi1/0 --bs 1 --epoch 1 --test_size 0.50",
    "python train_unet.py --device cuda:1 --exp_id levi1/1 --bs 1 --epoch 2 --test_size 0.50",
    "python train_unet.py --device cuda:1 --exp_id levi1/2 --bs 1 --epoch 3 --test_size 0.50",
    "python train_unet.py --device cuda:1 --exp_id levi1/3 --bs 1 --epoch 4 --test_size 0.50",
    "python train_unet.py --device cuda:1 --exp_id levi1/4 --bs 1 --epoch 5 --test_size 0.50",
    "python train_unet.py --device cuda:1 --exp_id levi1/5 --bs 1 --epoch 6 --test_size 0.50",
    "python train_unet.py --device cuda:1 --exp_id levi1/6 --bs 1 --epoch 7 --test_size 0.50",
    "python train_unet.py --device cuda:1 --exp_id levi1/7 --bs 1 --epoch 8 --test_size 0.50",
    "python train_unet.py --device cuda:1 --exp_id levi1/8 --bs 1 --epoch 9 --test_size 0.50",

    "python train_unet.py --device cuda:0 --exp_id levi2/0 --bs 1 --epoch 1 --test_size 0.30",
    "python train_unet.py --device cuda:0 --exp_id levi2/1 --bs 1 --epoch 2 --test_size 0.30",
    "python train_unet.py --device cuda:0 --exp_id levi2/2 --bs 1 --epoch 3 --test_size 0.30",
    "python train_unet.py --device cuda:0 --exp_id levi2/3 --bs 1 --epoch 4 --test_size 0.30",
    "python train_unet.py --device cuda:0 --exp_id levi2/4 --bs 1 --epoch 5 --test_size 0.30",
    "python train_unet.py --device cuda:0 --exp_id levi2/5 --bs 1 --epoch 6 --test_size 0.30",
    "python train_unet.py --device cuda:0 --exp_id levi2/6 --bs 1 --epoch 7 --test_size 0.30",
    "python train_unet.py --device cuda:0 --exp_id levi2/7 --bs 1 --epoch 8 --test_size 0.30",
    "python train_unet.py --device cuda:0 --exp_id levi2/8 --bs 1 --epoch 9 --test_size 0.30",
]

def main():
    
    bashcode = ''
    python_commands = exp_h

    commands = {
        f"new-{i+4}":bashcode + el for i,el in enumerate(python_commands)
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