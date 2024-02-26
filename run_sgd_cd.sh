for data in synth house wood casp bs
do
  for alg in sgd cd
  do
    for nu in 0.5 1.5 2.5 10 100
    do
      sbatch --array=0-100 start_sgd_cd.sh nu=$nu data=\"$data\" alg=\"$alg\"
    done
  done
done
