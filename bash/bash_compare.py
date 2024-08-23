import os

drug_list = ['RIF', 'INH', 'EMB', 'PZA', 'SM', 'OFX', 'CAP', 'AK', 'KAN', 'MOX', 'CIP']


# model_list = ['DeepAMR', 'CNNGWP', 'WDNN']


# for i in range(len(model_list)):
#     f = open(f'air{i}.sh', 'w')
#     f.write(f'#!/bin/bash\n')
#     f.write(f'conda run -n dataset-distillation python ../codes/compare_model.py --model {model_list[i]} \n')
#     f.close()
#     os.system(f'sbatch -n1 --gres=gpu:1 -o compare{i}.stdout ./air{i}.sh')
#     os.system(f'rm air{i}.sh')

# Update on 23 Aug 2024
model_list = ['MutEmbAblation_max', 'MutEmbAblation_mean', 'MutEmbAblation_sum']

for i in range(len(model_list)):
    f = open(f'air{i}.sh', 'w')
    f.write(f'#!/bin/bash\n')
    f.write(f'conda run -n dataset-distillation python ../codes/compare_model.py --model {model_list[i]} \n')
    f.close()
    os.system(f'sbatch -n1 --gres=gpu:1 -o compare{i}.stdout ./air{i}.sh')
    os.system(f'rm air{i}.sh')