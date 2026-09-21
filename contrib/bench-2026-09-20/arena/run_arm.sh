#!/bin/bash
# usage: run_arm.sh <run_id> <kscope|kscope_none> <domain:id>...
# Starts a memory server and an env server on free loopback ports, runs the driver, kills both.
set -uo pipefail
BASE=/path/to/kaleidoscope/experiments/bench-2026-09-20
RUN_ID=$1; ARM=$2; shift 2
PY=$BASE/venv-arena/bin/python
export BENCH_SCOPE=arena BENCH_SCOPE_CAP_USD=9 BENCH_RUN_ID=$RUN_ID BENCH_TAG="$RUN_ID/$ARM"
export HF_HOME=$BASE/cache/hf HF_HUB_CACHE=$BASE/data/.hfcache HF_HUB_DISABLE_TELEMETRY=1
export TIKTOKEN_CACHE_DIR=$BASE/cache/tiktoken PYTHONDONTWRITEBYTECODE=1
mkdir -p $BASE/logs $BASE/cache/hf $BASE/cache/tiktoken

read MEM_PORT ENV_PORT < <($PY -c 'import socket
s=[socket.socket() for _ in range(2)]
[x.bind(("127.0.0.1",0)) for x in s]
print(*[x.getsockname()[1] for x in s])')

(cd $BASE/MemoryArena/memory && exec $PY -m uvicorn server:app --host 127.0.0.1 --port $MEM_PORT --log-level warning) \
  >> $BASE/logs/arena_memserver-$RUN_ID-$ARM.log 2>&1 &
MEM_PID=$!
(cd $BASE/MemoryArena && exec $PY -m uvicorn env.env_server:app --host 127.0.0.1 --port $ENV_PORT --log-level warning) \
  >> $BASE/logs/arena_envserver-$RUN_ID-$ARM.log 2>&1 &
ENV_PID=$!
cleanup() { kill $MEM_PID $ENV_PID 2>/dev/null; wait $MEM_PID $ENV_PID 2>/dev/null; echo "servers stopped (mem pid $MEM_PID, env pid $ENV_PID)"; }
trap cleanup EXIT

for i in $(seq 1 60); do
  if curl -fsS -o /dev/null http://127.0.0.1:$MEM_PORT/openapi.json 2>/dev/null && \
     curl -fsS -o /dev/null http://127.0.0.1:$ENV_PORT/env/available 2>/dev/null; then break; fi
  if ! kill -0 $MEM_PID 2>/dev/null || ! kill -0 $ENV_PID 2>/dev/null; then echo "a server died during start-up"; exit 5; fi
  /bin/sleep 0.5
done
echo "memory server :$MEM_PORT (pid $MEM_PID)  env server :$ENV_PORT (pid $ENV_PID)  run_id=$RUN_ID arm=$ARM"

$PY $BASE/arena/run_smoke.py --run-id "$RUN_ID" --arm "$ARM" \
  --mem-url http://127.0.0.1:$MEM_PORT --env-url http://127.0.0.1:$ENV_PORT --tasks "$@"
RC=$?
echo "driver exit code $RC"
exit $RC
