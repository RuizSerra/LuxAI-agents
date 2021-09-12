#!/usr/bin/env bash

set -e

# Run from LuxAI-agents repo top level

WORKING_DIR="/Users/jaime/Downloads"

DATE=$(date '+%Y%m%d-%H%M%S')
TARGET_DIR_NAME="lux-${DATE}"

mkdir "${WORKING_DIR}/${TARGET_DIR_NAME}"
cp -r agents lux __init__.py agent_loader.py main.py "${WORKING_DIR}/${TARGET_DIR_NAME}"
cd $WORKING_DIR
tar -czf submission.tar.gz $TARGET_DIR_NAME

echo
echo "Packaged. You may submit with:"
echo "kaggle competitions submit -c lux-ai-2021 -f $WORKING_DIR/submission.tar.gz -m 'Message'"
echo
echo "Consider deleting ${WORKING_DIR}/${TARGET_DIR_NAME}"
