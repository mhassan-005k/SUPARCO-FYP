# Notebooks

Kaggle/Jupyter experiments:

- `faithsr_da_vs_base_stress_test.ipynb`: evaluates Base and degradation-adaptive checkpoints across bicubic, mild, medium, and severe synthetic degradations on 420 validation images.
- `faithsr_single_image_test.ipynb`: loads the Base checkpoint and produces a single-image qualitative/metric comparison.

The notebooks currently use hard-coded Kaggle dataset/model paths. They should be converted to configuration-driven paths before being treated as a reproducible evaluation pipeline.
