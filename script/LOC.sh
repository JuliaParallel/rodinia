#!/usr/bin/env bash

diff -y --suppress-common-lines $1 $2 | grep '^' | wc -l
