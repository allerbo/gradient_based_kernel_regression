data=$1
if [[ ${data:0:2} == "bs" ]]
then 
  first_seed=1
  last_seed=366
else
  first_seed=0
  last_seed=100
fi


for nu in 0.5 1.5 2.5 10 100
do
  sbatch --array=$first_seed-$last_seed start_sgd_cd.sh nu=$nu data=\"$data\"
done
