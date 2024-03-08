py -3.11 -m torch.distributed.run ^
--nproc_per_node 1 ^
--nnodes 2 ^
--node_rank 0 ^
-m mylib.train ^
--home_dir . ^
--conf "train.ini" ^
--task "ner"
