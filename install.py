import pip

pip.main(['install', 'git+https://github.com/huggingface/diffusers'])

# Remove Diffusers Version Requirement
filename = 'requirements.txt'

with open(filename, 'r') as file:
    lines = file.readlines()

with open(filename, 'w') as file:
    for line in lines:
        if line.startswith('diffusers=='):
            line = 'diffusers\n'
        file.write(line)
