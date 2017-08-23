#!/usr/bin/env bash
MY_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -eu

LOG_DIR="rotnet/basic"

export GOOGLE_APPLICATION_CREDENTIALS=${MY_DIR}/service_account.json

DAY=`date +"%Y%m%d_%Hh%M"`

RUN_ID=${DAY}_$(cat /dev/urandom | LC_CTYPE=C tr -dc 'a-z0-9' | fold -w 8 | head -n 1)
if [ ! -z ${1+x} ]; then RUN_ID=${RUN_ID}_"$1"; fi

cat >${MY_DIR}/run.sh <<EOF
#!/usr/bin/env bash

MY_DIR="\$( cd "\$( dirname "\${BASH_SOURCE[0]}" )" && pwd )"

source \${HOME}/.local/bin/virtualenvwrapper.sh
mktmpenv --no-cd -r \${MY_DIR}/requirements.txt

cd \${MY_DIR}

export GOOGLE_APPLICATION_CREDENTIALS=\${MY_DIR}/service_account.json
export TRAINING_ARTIFACTS=\${HOME}/training-artifacts/

python -u train/train_mnist.py \
 --log_dir=$LOG_DIR \
 --run_id=${RUN_ID}

deactivate

EOF
chmod a+x ${MY_DIR}/run.sh
PARAMS=$(cat ${MY_DIR}/run.sh)
leaderboard.py enq --log-dir="$LOG_DIR" --run-id="$RUN_ID" \
    --params="$PARAMS" --code-directory="${MY_DIR}"
