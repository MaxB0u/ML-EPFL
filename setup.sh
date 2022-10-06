# Setup script for installing dependencies on the local development environment to test

echo "Setting up local development environment with required libraries"
sudo update && sudo apt-get install -y python3-pip
sudo apt-get install -y git-lfs
git lfs install
pip3 install numpy
echo "Installed the required dependencies"
