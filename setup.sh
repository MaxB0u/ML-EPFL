# Setup script for installing dependencies on the local development environment to test

echo "Setting up local development environment with required libraries"
sudo update && sudo apt-get install -y python3-pip python3-tk python3-pytest
sudo apt-get install -y git-lfs
git lfs install

python3 -m pip install numpy
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade Pillow
python3 -m pip install setuptools
python3 -m pip install matplotlib
python3 -m pip install black
python3 -m pip install gitpython

echo "Formatting the python source code files according to black formatting"
echo "PS: https://github.com/psf/black"

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
black "$DIR"/

echo "Formatting done.."
echo "Local development environment setup done."