# FaithSR Updated Implementation Plan and Evidence-Driven Next Direction

Status: Native-3x core, native-x4 Swin2SR diagnostic and FaithSR-Lite-X4 ablation complete; linked EuroSAT multispectral TIFF experiment is next; restricted validation remains externally gated

Last updated: 2026-09-14

## 1. Final product requirement

System ka input 8-bit `.tiff` image hoga aur output **minimum 3x** spatial resolution ka 8-bit `.tiff` hoga. Current FaithSR pipeline native 3x hai, lekin native 4x Swin2SR bhi requirement-compliant hai aur usay sirf requirement ke liye 3x par resize karna zaroori nahi. Available TIFF/GeoTIFF metadata output ke sath preserve kiya jayega. Width, height, affine transform aur pixel spacing jaisi scale-dependent values declared output scale ke mutabiq update hongi; CRS, geographic bounds, band order, tags, no-data aur masks jahan valid hon wahan same rahenge.

Primary objective visually dramatic sharpening nahi, balkeh data-faithful reconstruction aur minimum hallucination hai. Model ko aisi building, road, edge ya texture invent nahi karni jo input observation support na karti ho.

## 2. Proposed research model

Working name: **FaithSR**; final architecture/name diagnostic evidence ke baad freeze hoga.

Native-x4 diagnosis aur us ke baad four-way FaithSR-Lite ablation dono complete hain. Fully fine-tuned Swin2SR strong fidelity reference hai lekin slow hai. Lite model latency pass karta hai, magar Full/Base quality result mixed-negative hai; is liye kisi variant ko final model freeze nahi kiya gaya. Agla gate linked EuroSAT multispectral TIFF data se domain/band-aware learning test karna hai. Mamba/global-context branch abhi add nahi hogi kyun ke long-range structure dominant measured failure prove nahi hua.

Implemented research components ye hain:

1. **Degradation encoder:** har input tile se blur, noise, compression aur aliasing ka compact representation estimate karega.
2. **Degradation-conditioned backbone:** estimated degradation ke mutabiq feature processing adapt karega.
3. **Native reconstruction heads:** PixelShuffle-compatible x3 and x4 heads respectively 512x512 tile ko 1536x1536 ya 2048x2048 banate hain.
4. **Learned observation operator:** SR output ko estimated degradation ke through dobara LR space mein project karega.
5. **Fidelity correction:** agar re-degraded output original input se disagree kare to unsupported detail ko reduce karega.
6. **Uncertainty head:** x3 experiments mein risk map test hui, lekin native-L2A calibration fail hui; isay validated probability claim nahi kiya jayega.
7. **Teacher distillation:** personalized pretrained teacher se transfer test hua, lekin x4 matched result near-neutral tha; is component ko proven gain nahi kaha jayega.

Fine-tuning a pretrained network valid personalization hai, magar paper contribution ko honestly distinguish kiya jayega. Native-x4 fine-tuned Swin2SR quality ceiling aur Lite Base speed floor hai. Next contribution band/domain evidence se select hogi, sirf architecture novelty ke liye module add nahi hoga.

GAN aur diffusion ko primary model mein use nahi kiya jayega, kyunke perceptual realism ke liye invented detail scientific interpretation ko damage kar sakti hai.

## 3. Dataset strategy

### Stage A - UC Merced

UC Merced sirf fast smoke tests aur existing results se continuity ke liye use hoga. Is se final scientific claim nahi banega.

### Stage B - Sentinel-2 Level-2A

Main public training/evaluation Sentinel-2 par hogi. Controlled RGB experiment ke liye B4/B3/B2 10 m bands HR target honge. Inko realistic blur, sensor noise, quantization aur exact 3x decimation se synthetic 30 m LR banaya jayega. Split random patches par nahi, geographic tiles aur acquisitions par hoga taa-ke leakage na ho.

Linked Kaggle EuroSAT collection ke 27,000 multispectral TIFF tiles broad ten-class pretraining aur downstream testing ke liye use honge. Parallel 27,000 JPG files sirf RGB control ke liye hain. Kyun ke tiles 64x64 hain, native Sentinel-2 L2A scenes 512x512 latency, large-raster behavior, scene-separated stress aur metadata audit ke liye primary rahenge. TIFFs ke exact dtype, 13-band order aur georeferencing ko training se pehle file-level audit confirm karega.

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
- pretrained teacher ko reproducible frozen/fine-tuned baseline banana; earlier x3 protocol parity aur new native-x4 product path ko separately label karna.

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

Benchmark target: confirmed hardware par 512x512 input to at least 1536x1536 output. Native-x4 ke liye output 2048x2048 hoga. Model-only aur complete TIFF read/tile/stitch/write latency separately report hogi. Warm-up, CUDA synchronization, at least 100 runs, mean/median/P95 aur peak VRAM record hongi.

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

### Phase 8 - Strongest-baseline diagnosis and justified hybrid decision

Completed v2 experiment ne fully fine-tuned Swin2SR ko native x4 par test kiya; x4 output ko x3 resize nahi kiya gaya. Frozen, scene-disjoint protocol par ye record hua:

- 512x512 to 2048x2048 model-only aur end-to-end TIFF mean/median/P95 latency, peak VRAM;
- degradation-family/severity-wise PSNR, SSIM, LPIPS, false-edge, edge precision/recall, SAM, ERGAS aur observation consistency;
- deterministic worst/median/best previews with metric JSON;
- failure categories: blur/noise/compression sensitivity, invented edges, broken long-range roads/fields, aur weak high-frequency detail.

Measured outcome:

- overall stress PSNR delta +0.5038 dB (95% scene-cluster CI +0.4019 to +0.6056) versus bicubic;
- false-edge delta -0.01460 and LPIPS delta -0.08691;
- compression-heavy PSNR delta +0.0193 dB with CI crossing zero;
- 512-to-2048 FP32 model-only P95 3.419 s and end-to-end P95 3.791 s on Tesla P100;
- uint8, CRS, tags, band descriptions, mask and recalculated x4 transform preserved.

Iske baad component selection evidence-driven hai:

- degradation-specific failure ho to degradation conditioning;
- hallucination/observation mismatch ho to bounded consistency correction;
- local frequency/detail failure ho to frequency refiner;
- long-range structure failure ho to lightweight Mamba/global-context branch;
- koi material failure na ho aur latency pass ho to unnecessary hybrid add nahi ki jayegi.

Exit criteria: complete. Deficiency ranking, confidence intervals, latency and deterministic visuals save hain. Next matched ablation: lightweight native-x4 base; plus personalized-teacher distillation; plus compression conditioning/consistency; full combination. Architecture novelty sirf quality-latency Pareto improvement par claim hogi.

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

Phases 0-9 execute ho chuki hain. Phase 9 ke Base, Distilled, Compression aur Full variants sab 60 epochs chale. Latency gate pass hua, magar registered quality gate fail/mixed raha: distillation near-neutral, compression LPIPS mein better lekin overall PSNR/observation mein worse, aur Full overall Base ko beat nahi karta.

Phase 10 execution order:

1. linked EuroSAT TIFF branch ka band/dtype/metadata/hash audit;
2. leakage-safe split manifest;
3. RGB B4/B3/B2 control against current Base and personalized Swin2SR;
4. explicit 13-band or selected-band model contract and mixed-resolution alignment;
5. EuroSAT pretraining followed by native-L2A scene evaluation;
6. model selection on fidelity-latency Pareto evidence;
7. restricted SUPARCO holdout as the final external-validity gate.
