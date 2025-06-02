#!/usr/bin/bash
#config: --utf-8

task_name="Isaac-Velocity-Flat-Lus1-vst"
load_run=""
logfile="log_$(date +%Y%m%d_%H%M%S).txt"

task_name="Flat-Lus2"
experiment_name="lus2_flat"
while getopts "n:l:c:e:tpsdar" arg; do
  case $arg in
    n)
      task_name=$OPTARG
      ;;
    t)
      run_mode="train"
      ;;
    p)
      run_mode="play"
      ;;
    s)
      run_mode="sim"
      ;;
    a)
      run_mode="assess"
      ;;
    l)
      load_run="--teacher_policy.resume_path $OPTARG"
      ;;
    c)
      checkpoint="--checkpoint $OPTARG"
      ;;
    d)
      play_demo="--play_demo_traj"
      ;;
    r)
      export_rknn="--export_rknn"
      ;;
    e)
      experiment_name=$OPTARG
      ;;
    ?)
      echo "Unknown args"
      exit 1
      ;;
  esac
done

    timestamp=$(date +'%Y-%m-%d-%H-%M-%S')
    echo "time is : $timestamp"

if [[ "$run_mode" == *"train"* ]]; then

    logfile="./logs/teacher/train_${task_name}_${idx}_${timestamp}.log"
    echo "log file is at $logfile"
    nohup python -u ./scripts/rsl_rl/train_teacher_policy.py --num_envs 100 $load_run  --headless > "$logfile" 2>&1 &
    tail -f $logfile

elif [[ "$run_mode" == *"play"* ]]; then
    logfile="./logs/st_rl/play_${task_name}_${idx}_${timestamp}.log"
    echo "log file is at $logfile"
  nohup python -u ./scripts/st_rl/play.py --task $task_name  $load_run $checkpoint $otherarg $play_demo --cfg_file load_run --experiment_name lus2_flat > "$logfile" 2>&1 &
    tail -f $logfile

elif [[ "$run_mode" == *"sim"* ]]; then
    echo "mujoco simulation.."
  python ./scripts/st_rl/sim2mujoco.py --task $task_name $load_run --experiment_name $experiment_name $export_rknn
elif [[ "$run_mode" == *"assess"* ]]; then
    echo "eval ..."
  python ./scripts/st_rl/eval_deploy.py --task $task_name $load_run --experiment_name $experiment_name
fi





