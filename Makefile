.PHONY: train-remote cuda-check

# Override GPU if 4090 is out of stock, e.g.:
#   GPU_ID="NVIDIA A40" DATA_CENTER_IDS=EU-SE-1 make cuda-check
train-remote:
	./scripts/remote_train.sh "$(CMD)"

cuda-check:
	./scripts/remote_train.sh "python scripts/hardware_probe.py --bench"