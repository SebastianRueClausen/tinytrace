#!/bin/sh

LINE_COUNT_THRESHOLD=7500

# 1. Get files tracked by git.
# 2. Exclude files that include "test".
# 3. Exclude files that aren't Rust code.
# 4. Read files.
# 5. Remove empty lines and comments.
# 6. Count lines.
LINE_COUNT=$(git ls-files | grep -v test | grep .rs | xargs cat | sed '/^\s*$/{d;};/^\s*\/\//{d;}' | wc -l)

if [ $LINE_COUNT -gt $LINE_COUNT_THRESHOLD ]; then
  echo "line count exceeds $LINE_COUNT_THRESHOLD"
  exit 1
fi
