import os
from packaging import version
import pkg_resources
from modules import errors
from modules.paths_internal import script_path
import subprocess, pip

# Get diffusers>=0.17.1 to add Kandinsky pipeline support
filename = os.path.join(script_path, 'requirements.txt')

target_version = version.parse('0.17.1')
package_name = 'diffusers'


if os.path.isfile(filename):
    print(f"Checking {package_name} version in requriments.txt")
    with open(filename, 'r') as file:
        lines = file.readlines()

    corrent_version_in_requirements = True
    with open(filename, 'w') as file:
        for line in lines:
            package_equals = "=="
            if line.startswith(f'{package_name}==') or line.startswith(f'{package_name}~='):
                if line.startswith(f'{package_name}~='):
                    package_equals = "~="
                else:
                    package_equals = ">="

                version_str = line[len(package_name) + 2:]
                if version_str != "":
                    current_version = version.parse(version_str)
                    print(f"Incompatible {package_name} version {current_version} in requirements.txt")
                    if current_version < target_version:
                        corrent_version_in_requirements = False
                        line = f'{package_name}{package_equals}{target_version}\n'
                        print(f"Changed {package_name} version to {package_equals}{target_version} in requirements.txt")

            file.write(line)

    if corrent_version_in_requirements:
        print(f"Correct {package_name} version in requriments.txt")

try:
    current_version = version.parse(pkg_resources.get_distribution(package_name).version)
    print(f'Current {package_name} version: {current_version}')

    if current_version < target_version:
        subprocess.run(['pip', 'install', f'{package_name}>={target_version}'])
        print(f'{package_name} upgraded to version {target_version}.')
        errors.print_error_explanation('RESTART AUTOMATIC1111 COMPLETELY TO FINISH INSTALLING PACKAGES FOR kandinsky-for-automatic1111')
    else:
        print(f'{package_name} is already up to date.')

except pkg_resources.DistributionNotFound:
    subprocess.run(['pip', 'install', f'{package_name}>={target_version}'])
    print(f'{package_name} installed with version {target_version}.')
