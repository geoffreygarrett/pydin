#!/bin/bash

# Install and freeze requirements
# Args:
# $1 - Python executable name
# $2 - requirements input file
# $3 - lock file output path
# $4 - windows format output path
install_and_freeze() {
    # Create and activate a virtual environment
    $1 -m venv venv
    source venv/bin/activate

    # Install requirements from the input file
    pip install -r $2

    # Freeze installed packages into the lock file
    pip freeze > $3

    # Convert \n to ; for windows and save it
    tr '\n' ';' < $3 > $4

    # Deactivate and remove the virtual environment
    deactivate
    rm -rf venv
}

# Define the requirements files
REQUIREMENTS_FILES=(
    "requirements_build.txt"
    "requirements_test.txt"
    "requirements_run.txt"
)

# Define the Python versions
PYTHON_VERSIONS=("python3.7" "python3.8" "python3.9")

# Process each requirements file
for REQ_FILE in "${REQUIREMENTS_FILES[@]}"; do
    # Generate the corresponding lock filename
    LOCK_FILE="${REQ_FILE/.txt/-lock.txt}"

    # Process each Python version
    for PYTHON in "${PYTHON_VERSIONS[@]}"; do
        # Generate the corresponding windows filename
        WINDOWS_FILE="${REQ_FILE/.txt/-windows-${PYTHON#python}.txt}"

        # Install and freeze the requirements
        install_and_freeze $PYTHON $REQ_FILE $LOCK_FILE $WINDOWS_FILE
    done
done
