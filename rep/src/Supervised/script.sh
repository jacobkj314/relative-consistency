make=allenai/
model=unifiedqa-v2-t5-large-1251000
use_deepspeed=false
is_checkpoint=false

python dataBundler.py -fa #-use_test #to get test results instead of dev results
echo BEGINNING RUN_SING ; bash run_single_unifiedqa.sh $make $model ../../../out/ $use_deepspeed $is_checkpoint; echo COMPLETED RUN_SING #train
echo BEGINNING COMPUTE_STATS ; bash compute_unifiedqa_stats.sh $model ../../../out/ $use_deepspeed ; echo COMPLETED COMPUTE_STATS  #evaluate
