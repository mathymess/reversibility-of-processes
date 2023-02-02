#!/bin/bash
announce() {
    echo -e '\033[1m' '\033[91m' "--------- $1" '\033[0m'
}

announce "Mypy:"
mypy --ignore-missing-imports *.py

announce "flakes8:"
flake8 --show-source --max-line-length 100 *.py

unset -f announce
