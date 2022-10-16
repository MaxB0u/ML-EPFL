# Script for running all tests under test/test_project1_public.py
# Run the ./setup.sh script before this to ensure all required dependencies have been installed in the testing environment

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" 

black "$DIR"/
pytest-3 --github_link "$DIR"/