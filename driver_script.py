import os

if __name__ == "__main__":
    for i in [0]:
        os.system(f'python main.py --exp {i} --gpu --exp_type self_train')
