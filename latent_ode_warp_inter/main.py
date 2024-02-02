# Main to run full commands and follow the debug section.

import subprocess

# Class
if __name__ == '__main__':
    #subprocess.run(
    #    ['python3', 'run_models.py', '--niters', '2500', '-n', '30', '-l', '10', '-s', '500', '--dataset', 'warp',
    #     '--latent-ode', '--lr', '0.015', '--noise-weight', '0.01',
    #     #'--rec-dims', '50', '--rec-layers', '3', '--gen-layers', '5', '--units', '50', '--gru-units', '50',
    #     '--viz'])
    # subprocess.run(['python3', 'run_models.py', '--niters', '100', '-n', '10', '-l', '10', '--dataset', 'warp', '--latent-ode',
    #            '--rec-dims', '2', '--rec-layers', '2', '--gen-layers', '2', '--units', '10', '--gru-units', '10', '--viz'])
    #subprocess.run(
    #     ['python3', 'run_models.py', '--niters', '100', '-n', '300', '-l', '20', '--dataset', 'physionet', '--latent-ode',
    #      '--rec-dims', '40', '--rec-layers', '3', '--gen-layers', '3', '--units', '50', '--gru-units', '50',
    #      '--quantization', '0.016', '--classif'])
    #subprocess.run(['python3', 'run_models.py', '--niters', '3000', '-n', '10', '-s', '300', '--dataset', 'warp',
    #                '--latent-ode', '--noise-weight', '0.01', '--viz'])
    #subprocess.run(['python3', 'run_models.py', '--niters', '3000', '-n', '10', '-s', '400', '--dataset', 'warp',
    #                '--latent-ode', '--noise-weight', '0.01', '--viz'])
    #subprocess.run(['python3', 'run_models.py', '--niters', '3000', '-n', '15', '-s', '300', '--dataset', 'warp',
    #                '--latent-ode', '--noise-weight', '0.01', '--viz', '--extrap', '--load', '77615'])
    #subprocess.run(['python3', 'run_models.py', '--niters', '3000', '-n', '10', '-s', '300', '--dataset', 'warp',
    #                '--latent-ode', '--noise-weight', '0.01', '--viz', '--load', '93187'])
    subprocess.run(['python3', 'run_models.py', '--niters', '3000', '-n', '10', '--dataset', 'warp',
                   '--latent-ode', '--noise-weight', '0.01', '--viz', '--load', '61672'])