for pid in `ps -edf | grep "/root/anaconda3/bin/jupyter-notebook --no-browser" | grep "python" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "prover_env.py" | grep "python" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "ray/workers/default_worker.py" | grep "python" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "ray/core" | grep "python" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "ray/plasma" | grep "python" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "ray/global_scheduler" | grep "python" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "python main.py" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "python -u main.py" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "multiprocessing.spawn" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "python -u training_examples_collection.py" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "python -c from multiprocessing" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "python" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "beagle" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
for pid in `ps -edf | grep "eprover" | grep -v grep | awk '{print $2}'`; do kill -9  $pid ; done
rm -rf /tmp/eProver_out_*
rm -rf /tmp/plasma_* /tmp/ray*
