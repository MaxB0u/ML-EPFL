# Script for running all tests under test/test_project1_public.py
# Run the ./setup.sh script before this to ensure all required dependencies have been installed in the testing environment

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" 

black "$DIR"/
TEST_OUTPUT=$(pytest-3 --github_link "$DIR"/)
FAILURE_PATTERN="failed"

if [[ $TEST_OUTPUT == *"$FAILURE_PATTERN"* ]];
then
    echo "[ERROR] TEST FAILURE(s) DETECTED"
    error_log=$(echo $TEST_OUTPUT | tail -c 100)
    echo "[INFO] Error Log: "
    echo $error_log
else
    echo "[INFO] ALL TESTS PASSED"
fi