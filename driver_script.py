import os

if __name__ == "__main__":
    for i in range(1):
        os.system(f'python main.py --exp {i} --exp_type cifar10 --gpu --gpu_type a100')
