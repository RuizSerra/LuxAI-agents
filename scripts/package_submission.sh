#!/usr/bin/env bash

set -e

# Run from LuxAI-agents repo top level

WORKING_DIR="/Users/jaime/Downloads"

echo "Working dir: " $WORKING_DIR

DATE=$(date '+%Y%m%d-%H%M%S')
TARGET_DIR_NAME="lux-${DATE}"

echo "Target dir: " "${WORKING_DIR}/${TARGET_DIR_NAME}"

mkdir "${WORKING_DIR}/${TARGET_DIR_NAME}"
cp -r agents agent_loader.py main.py "${WORKING_DIR}/${TARGET_DIR_NAME}"
cd $WORKING_DIR
tar -czf submission.tar.gz $TARGET_DIR_NAME
