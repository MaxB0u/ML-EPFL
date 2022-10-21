# Script for running all tests under test/test_project1_public.py
# Run the ./setup.sh script before this to ensure all required dependencies have been installed in the testing environment

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )" 

black "$DIR"
TEST_OUTPUT="$(pytest-3 --github_link "$DIR")"
FAILURE_PATTERN="failed"
LOG_ALL_TEST_OUTPUTS=false
IS_GITHUB_ACTION=$1

if [ -z "$IS_GITHUB_ACTION" ];
then
    LOG_ALL_TEST_OUTPUTS=true # Only show full logs for local run
fi

if [[ $TEST_OUTPUT == *"$FAILURE_PATTERN"* ]];
then
    echo "[ERROR] TEST FAILURE(s) DETECTED"
    error_log="$(echo $TEST_OUTPUT | tail -c 100)"
    failed_tests="$(grep -E "test" <<< $TEST_OUTPUT)"

    echo "[INFO] Failed tests: "
    echo "$failed_tests"

    if [[ "$LOG_ALL_TEST_OUTPUTS" == true ]];
    then
        echo "[INFO] Full logs: "
        echo "$TEST_OUTPUT"
    fi

    echo "[INFO] Error Summary: "
    echo "$error_log"
else
    echo "[INFO] ALL TESTS PASSED"
fi