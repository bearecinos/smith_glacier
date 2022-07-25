#!/bin/bash

UNMUTE=$1
shift 1

if [ "$PMI_RANK" == "$UNMUTE" ]; then
  exec $*
else
  exec $* >/dev/null 2>&1
fi
