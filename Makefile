train:
	PYTHONPATH=src python src/cli/train.py

ddp:
	PYTHONPATH=src torchrun --nproc_per_node=3 src/cli/train.py

typecheck:
	mypy src/