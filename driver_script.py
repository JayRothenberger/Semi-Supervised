import os

if __name__ == "__main__":
    for i in [1]:
        os.system(f'python experiment.py --exp {i} --gpu --exp_type test')
