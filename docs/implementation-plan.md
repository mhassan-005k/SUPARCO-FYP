# FaithSR-3X Updated Implementation Plan

Status: Core implementation complete; preliminary EuroSAT training complete; Sentinel-2, full ablations, pretrained comparison, and restricted SUPARCO validation pending

Last updated: 2026-09-09

## 1. Final product requirement

System ka input 8-bit `.tiff` image hoga aur output exactly 3x spatial resolution ka 8-bit `.tiff` hoga. Available TIFF/GeoTIFF metadata output ke sath preserve kiya jayega. Width, height, affine transform aur pixel spacing jaisi scale-dependent values 3x output ke mutabiq update hongi; CRS, geographic bounds, band order, tags, no-data aur masks jahan valid hon wahan same rahenge.

Primary objective visually dramatic sharpening nahi, balkeh data-faithful reconstruction aur minimum hallucination hai. Model ko aisi building, road, edge ya texture invent nahi karni jo input observation support na karti ho.

## 2. Proposed research model

Working name: **FaithSR-3X**

Final model sirf pretrained model ka wrapper nahi hoga. Pretrained Swin2SR/HAT ko strong teacher aur baseline ke taur par use kiya jayega, jabke deploy hone wala student hamara custom lightweight network hoga:

1. **Degradation encoder:** har input tile se blur, noise, compression aur aliasing ka compact representation estimate karega.
2. **Degradation-conditioned backbone:** estimated degradation ke mutabiq feature processing adapt karega.
3. **Native 3x reconstruction head:** PixelShuffle-compatible 3x head 512x512 tile ko 1536x1536 banayega.
4. **Learned observation operator:** SR output ko estimated degradation ke through dobara LR space mein project karega.
5. **Fidelity correction:** agar re-degraded output original input se disagree kare to unsupported detail ko reduce karega.
6. **Uncertainty head:** har output region ka confidence/risk map dega, taa-ke analyst questionable reconstruction identify kar sake.
7. **Teacher distillation:** pretrained teacher se useful spatial knowledge transfer hogi, lekin student ki architecture, losses, consistency mechanism aur uncertainty hamari research contribution honge.

GAN aur diffusion ko primary model mein use nahi kiya jayega, kyunke perceptual realism ke liye invented detail scientific interpretation ko damage kar sakti hai.

## 3. Dataset strategy

### Stage A - UC Merced

UC Merced sirf fast smoke tests aur existing results se continuity ke liye use hoga. Is se final scientific claim nahi banega.

### Stage B - Sentinel-2 Level-2A

Main public training/evaluation Sentinel-2 par hogi. Controlled RGB experiment ke liye B4/B3/B2 10 m bands HR target honge. Inko realistic blur, sensor noise, quantization aur exact 3x decimation se synthetic 30 m LR banaya jayega. Split random patches par nahi, geographic tiles aur acquisitions par hoga taa-ke leakage na ho.

Sentinel-2 normally 16-bit scientific product hai, jabke final SUPARCO input 8-bit confirmed hai. Is domain gap ko explicitly handle karne ke liye:

- Sentinel reflectance ko pehle scientifically correct scale par read kiya jayega;
- training samples ke liye documented 8-bit mapping/quantization version banegi;
- native scientific values aur derived 8-bit inputs separate manifests mein record honge;
- final claim restricted 8-bit SUPARCO holdout validation ke baghair nahi ki jayegi.

### Stage C - SUPARCO on-site data

Public model ko frozen baseline ke taur par test kiya jayega. Permission milne par sirf compact adapters/degradation modules fine-tune honge. Frozen aur adapted models ko separate holdout par compare kiya jayega.

## 4. Implementation phases

### Phase 0 - Reproducible project foundation

- environment file aur pinned dependencies;
- central YAML experiment configuration;
- deterministic seeds and run IDs;
- dataset/checkpoint/output paths ko Kaggle-specific hard-coding se remove karna;
- every run ke metrics, config, Git commit, timing aur artifacts ka manifest;
- current v0.1 results ko immutable baseline rakhna.

Exit criteria: local aur Kaggle par same config se same pipeline start ho aur secrets Git mein na jayen.

### Phase 1 - TIFF/GeoTIFF I/O and metadata pipeline

- Rasterio-based 8-bit multi-band reader;
- ordinary TIFF aur GeoTIFF dono ke liye safe profile capture;
- overlapping window/tile reader;
- 3x writer with updated dimensions and transform;
- CRS, bounds, tags, band descriptions, color interpretation, no-data, masks and compatible compression preservation;
- sidecar machine-readable audit report showing copied, recalculated aur unavailable fields.

Exit criteria: synthetic fixtures par input/output metadata tests 100% pass hon; raster dimensions exactly 3x aur geographic bounds unchanged hon.

### Phase 2 - Native 3x baseline

- bicubic, bilinear aur nearest baselines;
- current FaithSR backbone ko 3x reconstruction head dena;
- same parameter budget mein plain lightweight CNN baseline;
- pretrained teacher ko 3x task ke liye reproducible inference/fine-tuning baseline banana.

Exit criteria: all models arbitrary valid tile sizes par correct 3x tensor shape dein aur evaluation script ek common metrics table generate kare.

### Phase 3 - Realistic degradation pipeline

- blur kernels: isotropic, anisotropic and MTF-inspired;
- Gaussian/Poisson noise;
- aliasing-aware 3x downsampling;
- 8-bit quantization, clipping and optional TIFF compression artifacts;
- easy-to-severe degradation curriculum;
- every LR sample ke exact degradation parameters manifest mein save karna.

Exit criteria: degradation generation deterministic/replayable ho aur train/validation/test geography isolated ho.

### Phase 4 - FaithSR-3X core research model

- improved degradation encoder;
- degradation-conditioned residual/attention blocks;
- learned re-degradation operator;
- observation-consistency correction with bounded strength;
- uncertainty prediction head;
- teacher-student feature/output distillation;
- lightweight parameter/FLOP target ke andar architecture search/ablation.

Training loss:

`L = L_charbonnier + L_ms-ssim + L_gradient + L_frequency + L_observation + L_distillation + L_uncertainty`

Weights validation set par tune hongi aur final test se pehle freeze hongi.

Exit criteria: matched base ke muqable statistically meaningful fidelity gain ya hallucination reduction; sirf selected images ki visual improvement acceptable evidence nahi hogi.

### Phase 5 - Hallucination and scientific-fidelity evaluation

Core metrics: PSNR, SSIM, LPIPS, re-degradation error, edge precision/recall aur false-edge rate. Multiband experiment mein SAM/ERGAS add honge. Downstream validation mein building/road/land-cover model ka mIoU/F1 compare hoga. Uncertainty ko error correlation aur calibration metrics se test kiya jayega.

Required comparisons:

- interpolation baselines;
- pretrained teacher without adaptation;
- pretrained teacher full fine-tuning;
- FaithSR plain base;
- base + degradation conditioning;
- base + learned consistency;
- base + distillation;
- base + uncertainty;
- complete FaithSR-3X.

Exit criteria: complete dataset results, confidence intervals, failure cases aur ablation table available ho.

### Phase 6 - Speed optimization

Benchmark target: confirmed hardware par 512x512 input to 1536x1536 output. Model-only aur complete TIFF read/tile/stitch/write latency separately report hogi. Warm-up, CUDA synchronization, at least 100 runs, mean/median/P95 aur peak VRAM record hongi.

Optimization order:

1. efficient student architecture;
2. FP16/AMP;
3. `torch.compile` where stable;
4. ONNX export;
5. TensorRT engine;
6. structured pruning/distillation;
7. INT8 sirf tab jab fidelity regression accepted threshold ke andar ho.

Exit criteria: sub-one-second claim sirf specified hardware aur clearly defined latency scope ke sath report ho.

### Phase 7 - Analyst application

- `.tiff` upload and validation;
- metadata summary before processing;
- 3x tiled inference with progress;
- original, bicubic and FaithSR comparison;
- uncertainty/confidence overlay;
- downloadable 8-bit TIFF/GeoTIFF;
- downloadable audit JSON containing model version, checksum, runtime, metadata handling and warnings.

Exit criteria: representative large raster without seams process ho, output reopen ho aur metadata verification pass ho.

## 5. Mandatory tests

- 8-bit dtype remains `uint8` after round trip;
- width and height are exactly multiplied by 3;
- output transform has one-third pixel spacing;
- CRS and bounds remain unchanged for GeoTIFF;
- tags, band order/descriptions, no-data and masks preserved where supported;
- unsupported/inconsistent metadata is reported, not silently invented;
- tiled and full-image inference agree within tolerance;
- overlap stitching creates no visible seams;
- output values are clipped safely to 0-255;
- deterministic mode returns repeatable output;
- corrupt file, unsupported band schema and memory-limit cases fail safely;
- re-degraded SR remains close to original LR observation.

## 6. Kaggle execution plan

Kaggle credentials ignored hi rahengi aur explicit approval se pehle koi remote execution nahi hogi. Approval ke baad workflow:

1. core package and tests repo mein complete karna;
2. thin Kaggle notebook se package install/import karna;
3. Sentinel-2 dataset/version pin karna;
4. baseline x3 run;
5. FaithSR-3X training with checkpoint resume;
6. ablation jobs;
7. metrics/artifacts download and audit;
8. verified results ko experiment log aur paper tables mein add karna.

## 7. Paper development alongside code

Har experiment ke sath config, hypothesis, dataset split, result, failure aur interpretation experiment log mein add honge. Paper sections parallel update hongi:

- Phase 1-3: dataset and methodology;
- Phase 4: proposed architecture and losses;
- Phase 5: results, ablations and hallucination analysis;
- Phase 6-7: efficiency and deployment;
- final SUPARCO test: limitations and external validity.

Claim tabhi paper mein final hoga jab corresponding experiment, metric implementation aur artifact reproducible ho.

## 8. Recommended execution order

Sab se pehle Phase 0 aur Phase 1 implement hon. Uske baad native 3x baseline aur realistic degradation dataset banaya jaye. Strong baseline establish hone ke baad FaithSR-3X ke components one-by-one add aur ablate hon. Model fidelity pass karne ke baad hi deployment optimization aur application UI final ki jaye.
