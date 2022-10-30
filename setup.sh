# Setup script for installing dependencies on the local development environment to test

echo "Setting up local development environment with required libraries"
sudo apt-get update && sudo apt-get install -y python3-pip python3-tk python3-pytest
sudo apt-get install -y git-lfs
git lfs install

python3 -m pip install --upgrade pip
python3 -m pip install numpy==1.23.1
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow
python3 -m pip install setuptools
python3 -m pip install matplotlib==3.5.2
python3 -m pip install black==21.4b0
python3 -m pip install gitpython==3.1.18
python3 -m pip install pytest-mock==3.7.0
python3 -m pip install pytest==7.1.2

# echo "Formatting the python source code files according to black formatting"
# echo "PS: https://github.com/psf/black"

# DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# black "$DIR"/

# echo "Formatting done.."
echo "Local development environment setup done."
