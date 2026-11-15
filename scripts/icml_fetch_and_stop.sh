#!/bin/bash
# Usage: icml_fetch_and_stop.sh SESSION REMOTE_FILE LOCAL_FILE ; polls until the remote file exists, downloads with retries, then stops SESSION.
S=$1; R=$2; L=$3; C=~/.local/bin/colab
for i in $(seq 1 120); do
  if timeout 90 $C ls -s $S $(dirname $R) 2>/dev/null | grep -q "$(basename $R)"; then
    for j in 1 2 3 4 5 6; do timeout 180 $C download -s $S $R $L >/dev/null 2>&1; [ -s $L ] && break; sleep 20; done
    break
  fi
  sleep 60
done
for k in 1 2 3; do timeout 120 $C stop -s $S >/dev/null 2>&1; sleep 5; timeout 60 $C sessions 2>/dev/null | grep -q "\[$S\]" || break; done
echo "fetched=$([ -s $L ] && echo yes || echo no) stopped=$(timeout 60 $C sessions 2>/dev/null | grep -q "\[$S\]" && echo no || echo yes)"
