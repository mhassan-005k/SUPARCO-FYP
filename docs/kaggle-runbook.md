# Kaggle Runbook

Remote execution is approved, but a job must not be submitted until the local test suite passes and `kaggle/kernel-metadata.json` contains the real account ID and attached versioned dataset slug.

1. Keep `kaggle.json` only at the repository root; it is ignored by Git.
2. Copy it to Kaggle CLI's expected local configuration directory with restricted file permissions.
3. Update the kernel `id` and `dataset_sources` without placing secrets in metadata.
4. Attach a dataset containing documented 8-bit derived Sentinel-2 images or approved public input.
5. Run `kaggle kernels push -p kaggle`.
6. Monitor status and download output only after completion.
7. Validate `run.json`, `split.json`, `metrics.csv`, `best.pth`, and `last.pth` before accepting the run.
8. Record the exact dataset version, kernel version, Git commit, GPU, runtime, and metrics in the experiment log.

Because the GitHub repository is private, the kernel uses the private `muhammadhassan2/faithsr-3x-private-source` Kaggle dataset. This dataset contains only `src/faithsr`, the selected YAML configuration, and `pyproject.toml`. It excludes credentials, checkpoints, documents, notebooks, results, and restricted imagery.

The configured first run uses `apollo2506/eurosat-dataset/EuroSAT`, the 8-bit RGB branch, only as a preliminary public baseline. Its 64x64 scenes and classification-oriented construction are not sufficient for the final scientific claim. The all-band TIFF branch and SEN2VENUS require separate bit-depth, storage, split, and licensing validation before use.
