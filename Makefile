train:
	PYTHONPATH=src python scripts/train.py --phase 1

ddp:
	PYTHONPATH=src torchrun --nproc_per_node=3 scripts/train.py --phase 1

typecheck:
	mypy src/