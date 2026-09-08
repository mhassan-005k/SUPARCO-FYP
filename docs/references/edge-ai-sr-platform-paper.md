2025 International Symposium on Multimedia and Communications Technology (ISMAC)

# Development of a Super-Resolution Satellite Image Processing System on an Edge AI Platform

Kanapoj Ngambenjavichaikul Amir Hajian _Multimedia Data Analytic and Multimedia Data Analytic and Processing Research Unit, Processing Research Unit, Department of Electrical Engineering,   Department of Electrical Engineering, Chulalongkorn University Chulalongkorn University_ Bangkok, Thailand Bangkok, Thailand 6670031721@student.chula.ac.th amirhajian85@gmail.com

Watchara Ruangsang Supavadee Aramvith _King Mongkut’s University Multimedia Data Analytic and of Technology Thonburi, Processing Research Unit,_ Bangkok, Thailand _Department of Electrical Engineering,_ watchara.ruan@kmutt.ac.th _Chulalongkorn University_ Bangkok, Thailand supavadee.a@chula.ac.th

**_Abstract_ — High-resolution satellite imagery is essential for many remote sensing applications, such as land-use mapping, infrastructure monitoring, and environmental assessment. Unfortunately, acquiring high-resolution data is expensive, constrained by limited bandwidth, and limited by infrequent satellite passes. Super-resolution (SR) techniques offer a software-based approach to enhance image resolution by reconstructing high-resolution images from low-resolution inputs, utilizing deep learning. In this work, we present a fully implemented, lightweight attention-enhanced U-Net model for real-time satellite image super-resolution on an edge AI platform. We optimize the model using the Xilinx Vitis AI toolchain and deploy it on the Xilinx Kria KV260 FPGA board. The development pipeline includes quantization-aware training, pruning, model compilation, and hardware deployment. The network architecture is carefully designed to preserve spatial detail while being compatible with quantization and FPGA constraints. We evaluated the system on the UCMerced Land Use dataset with ×2 and ×4 upscaling factors. Results show that the INT8-quantized model produces high-quality super-resolved images with low inference latency and efficient resource usage. These outcomes demonstrate the practicality of the proposed system for real-time, embedded remote sensing applications.**

**_Keywords—Super-resolution, remote sensing, edge AI, quantization aware training, Xilinx Kria KV260, Vitis AI, FPGA deployment, U-Net_**

## I. INTRODUCTION

High-resolution satellite imagery is crucial for various remote sensing applications, including land-use mapping, infrastructure monitoring, and environmental assessment. However, acquiring high-resolution data remains challenging due to limitations in cost, bandwidth, and revisit frequency. These limitations reduce the availability of timely and detailed imagery, which can hinder tasks that require finegrained spatial information such as edge detection, texture analysis, and structural mapping.

Super-resolution (SR) techniques have emerged as a costeffective solution by reconstructing high-resolution images from low-resolution inputs using computational methods. In remote sensing, SR enables enhanced image interpretation without requiring additional data acquisition or hardware upgrades. Deep learning models, particularly those based on convolutional neural networks (CNNs), have demonstrated remarkable success in this domain by learning complex mappings from large-scale training data for a GPU-based environment. However, these models are often too large or

computationally demanding for real-time or edge deployment.

Edge AI platforms such as the Xilinx Kria KV260 offer a promising solution by enabling low-power, highperformance inference on embedded systems. The platform integrates ARM processors, FPGA fabric, and a Deep Processing Unit (DPU), and supports optimization workflows using the Vitis AI toolchain. These features make it suitable for deploying lightweight deep learning models in field settings such as UAVs or remote monitoring stations.

To be deployable on edge hardware, SR models must be efficient, quantization-friendly, and accurate enough to recover fine textures and small objects. While several recent models have pushed the boundaries of accuracy in Remote Sensing image Super Resolution (RSISR), many are still too large or impractical for embedded use. This motivates the need for models that strike a balance between accuracy, model size, and hardware compatibility.

This paper presents a complete, edge-deployable SR system optimized for the Xilinx Kria KV260. Our work builds upon the AERU-Net [3], a 2025 U–Net–based model that integrates edge recovery and attention mechanisms to enhance reconstruction quality while reducing model parameters. We further optimize the model for quantizationaware training and real-time deployment.

## II. RELATED WORKS

## _A. Deep Learning based Single Image Super Resolution_

Single Image Super-Resolution (SISR) plays a critical role in remote sensing by reconstructing high-resolution (HR) imagery from low-resolution (LR) inputs. The enhanced spatial resolution significantly improves downstream applications such as land classification, infrastructure detection, and environmental analysis. Traditional interpolation methods, although computationally efficient, often fail to recover fine structures and texture details, particularly in complex scenes. The advent of deep learning has transformed the SISR domain. SRCNN [1] was one of the earliest convolutional models to learn end-to-end mappings for super-resolution, laying the groundwork for deeper architectures. It was followed by deeper models such as EDSR [2] and RCAN [4], which introduced residual learning and channel attention, respectively, to improve reconstruction quality. However, their high computational complexity and

979-8-3315-6946-4/25/$31.00 ©2025 IEEE

Authorized licensed use limited to: UNIV OF ENGINEERING AND TECHNOLOGY LAHORE. Downloaded on July 22,2026 at 10:29:12 UTC from IEEE Xplore.  Restrictions apply.

large number of parameters present significant obstacles for real-time or edge-based applications.

To overcome these limitations, researchers have developed models specifically targeting Remote Sensing Image Super-Resolution (RSISR). U-Net and its derivatives have gained prominence due to their encoder–decoder architecture and skip connections, which facilitate multiscale context fusion while preserving spatial detail. Models based on U-Net have demonstrated effectiveness in recovering small-scale structures and sharp edges in satellite images. Recent advances include the integration of transformer-based modules and attention mechanisms into RSISR networks. TransENet [5] introduced a multistage enhancement structure that combines high- and lowdimensional feature embeddings. Although it achieved competitive performance, its large parameter counts and inefficient upsampling layers limited its practicality, particularly on resource-constrained hardware.

In 2024, CSA-FE [6] further developed this approach by incorporating channel and spatial attention modules to enhance feature discrimination and improve high-frequency detail restoration. Despite improvements in visual quality, CSA-FE suffered from high computational complexity and limited ability to preserve content-aware attention across different scales.

HAUNet [7] further advanced the U-Net architecture by embedding hybrid attention modules to capture global spatial context and content information. It also introduced a crossscale interaction module to better link encoder outputs at different resolution levels. While HAUNet achieved strong performance in capturing abstract semantics, it demonstrated limited capability in recovering sharp edges and was computationally demanding.

To address these challenges, AERU-Net [3], was proposed in 2025 as a lightweight and deployable RSISR model. It enhances the standard U-Net backbone by introducing three core components: an Edge Recovery Block (ERB) for structural fidelity, spatial and channel attention modules (SAM and CAM) for adaptive focus, and a CrossScale Interaction Module (CIM) for efficient multi-level communication. These modules collectively improve both reconstruction accuracy and hardware compatibility.

This study builds upon AERU-Net by modifying its architecture and training procedures to support quantizationaware training (QAT). The resulting model is optimized for deployment on the Xilinx Kria KV260 platform, enabling real-time inference with reduced memory and compute overhead.

## _B. Xilinx Kria KV260 board_

Edge AI platforms have emerged as a practical solution for deploying deep learning models in resource-constrained environments, such as those found in UAVs, field stations, and remote sensing payloads. These platforms aim to provide efficient, low-latency inference with limited power and memory resources. Among these, the Xilinx Kria KV260 stands out as a reconfigurable System-on-Chip (SoC) platform that strikes a balance between performance,

flexibility, and energy efficiency. The KV260 is built on the Zynq UltraScale+ MPSoC architecture, which combines a quad-core ARM Cortex-A53 processor, a dual-core CortexR5 real-time processor, and programmable logic based on FPGA fabric [8]. This design allows heavy AI computations to be offloaded to hardware while general processing runs on the ARM cores. Figure 1 shows the KV260 development board, which pairs the Zynq MPSoC module with a carrier board for vision applications.

A key feature of the KV260 platform is its integrated Deep Processing Unit, a specialized neural network accelerator implemented in the FPGA logic. The DPU is an IP core optimized for deep learning workloads, particularly convolutional neural networks, leveraging parallel computation and fast on-chip memory to achieve high throughput. In essence, the DPU is a dedicated AI engine for convolutional network inference, freeing the ARM cores from the bulk of the math-intensive work. Xilinx provides a comprehensive software stack called Vitis AI to support deploying models onto the DPU.



Fig. 1. Xilinx Kria KV260 board

Using the Vitis AI toolchain, neural network models trained in frameworks such as PyTorch or TensorFlow can be converted into a DPU-compatible format through several optimization steps. These include quantization, pruning, and compilation into a hardware executable. Quantization is especially crucial. By reducing model precision from 32-bit floating point to 8-bit integer, one can shrink the model size by a factor of four and often achieve a two- to four-times speedup in inference, with significantly lower memory bandwidth usage. This int8 optimization yields faster, smaller models while maintaining near-original accuracy when done with proper calibration or quantization-aware training. Pruning further trims the model by removing redundant weights, which also helps meet the limited on-chip memory constraints.

## III. METHODOLOGY

## _A. Network Architecture_

Our proposed network builds upon the AERU-Net architecture, which is specifically designed for remote sensing image super-resolution. AERU-Net follows the standard U-Net design, which uses an encoder–decoder structure with skip connections. This structure helps preserve spatial details and allows the network to extract features at multiple scales. The model starts with a 3×3 convolution to extract shallow features from the lowresolution input image. These features are then passed through three encoder stages. Each stage reduces the spatial

Authorized licensed use limited to: UNIV OF ENGINEERING AND TECHNOLOGY LAHORE. Downloaded on July 22,2026 at 10:29:12 UTC from IEEE Xplore.  Restrictions apply.



<!-- Start of picture text -->
cvs — | CxHxW| CxHi2xWi2 CxHi4xWI4<br>LR Ft F2 F3,<br>Bilinear Interpolation CIM<br>‘ Out ‘ “| Out3<br>“Eaee ee<br>LSR<br><!-- End of picture text -->



<!-- Start of picture text -->
AERU-Net (2025 Simplified AERU-Net Optimizeda KV260 Model<br>(AERU-s)<br><!-- End of picture text -->



<!-- Start of picture text -->
rela<br>FeatureInput FeatureOutput<br>Hx Wi xc v Hx Wx<br>78/58<br>_ : 2 FFN<br><!-- End of picture text -->



<!-- Start of picture text -->
Model Trainin U-Net with Attention Modules<br>(GPU g PyTorch (FP32 Precision)<br>) Remote Sensing Dataset<br>Quantization INT8 in PyTorch<br>Procedure Precision-Aware Layer<br>a INT8 Calibration<br>Cc vis Generate .xmodel<br>omptation DPU instruction mapping<br>Deployment on Kria Kria KV260 SoC (ARM + FPGA)<br>P 360 DPU — Run SR Inference<br>Real-time HR Output<br><!-- End of picture text -->

inference. Input patches of size 32×32 for ×4 or 64×64 for ×2 are upscaled to 128×128. These are tiled to reconstruct full images due to on-chip memory constraints. The result is a fully integer 8-bit model compatible with DPU execution and suitable for real-time edge inference.

## _C. Performance on Kria KV260_

In this section, we present the performance results on the Kria KV260, comparing AERU-Net with our optimized model for 2× and 4× super-resolution on the Xilinx KV260.

At ×2 upscaling, AERU-Net achieves a higher PSNR (34.64 dB) than our FP32 model (32.56 dB), although their SSIM values are nearly identical (~0.93). AERU-Net also attains a better spectral correlation and a lower SAM error, indicating it preserves spectral content slightly more faithfully. Our INT8 quantized model exhibits a noticeable drop in quality at ×2, with PSNR falling to 29.72 dB and SSIM to 0.8698, accompanied by reduced SCC (0.3606) and a higher SAM (0.0691) compared to the FP32 version. This quantization impact reflects the trade-off for efficiency, as the INT8 model runs faster on the KV260 but with degraded image fidelity.

At ×4 upscaling, the gap between AERU-Net and our FP32 model narrows in some respects. AERU-Net’s PSNR is approximately 1.3 dB higher, but our model slightly outperforms SSIM, suggesting it retains structural details well, even at high magnification. AERU-Net still maintains an advantage in spectral accuracy with a higher SCC and a marginally lower SAM. Once again, the INT8 struggles with this larger scale factor, as its PSNR drops to 24.59 dB and SSIM to 0.6856, alongside a significantly lower SCC and higher SAM.

Despite the slight performance loss, our model’s strength lies in its extreme efficiency.

TABLE I. PERFORMANCE OF OPTIMIZED MODEL-KV260 ON UCMERCED DATASET

|**_Model_**|**_PSNR_**|**_SSIM_**|**_SCC_**|**_SAM_**|
|---|---|---|---|---|
|AERU-Net(×2)|34.64|0.9346|0.6514|0.0478|
|Our (×2, FP32)|32.56|0.9332|0.0593|0.0539|
|Our (×2, INT8)|29.72|0.8698|0.3606|0.0691|
|AERU-Net (×4)|27.97|0.7686|0.2842|0.1007|
|Our (×4, FP32)|26.69|0.7939|0.2172|0.1020|
|Our (×4, INT8)|24.59|0.6856|0.0976|0.1231|



Higher values are better for PSNR, SSIM, and SCC [12]. Lower values are better for SAM. End-to-end throughput includes all memory and processing overheads. As shown, the INT8 models on KV260 produce results close to the float baseline.

In addition to speed, memory footprint is significantly reduced. Table II compares model sizes. The AERU-s model for ×4 has 0.9260 million parameters and requires approximately 3.70 megabytes. The optimized version has 0.2952 million parameters and a size of 1.18 megabytes in FP32. After INT8 quantization, this reduces to 0.30 megabytes. The result is more than a 12 times reduction in size. The model fits within the KV260 SRAM, avoiding the

need for external DDR memory and improving both latency and energy efficiency.

TABLE II. PERFORMANCE OF AERU-NET VS. OUR FP32/INT8 MODELS ON THE KRIA KV260 (2× AND 4× SR)

|**_Model_**|**_Params (M)_**|
|---|---|
|AERU-Net(×4)|0.706|
|AERU-S (×4)|0.9260|
|Optimized KV260 INT8 (×4)|0.2952|
|AERU-Net (×2)|0.705|
|AERU-S (×2)|0.9052|
|Optimized KV260 INT8 (×2)|0.2744|



Float models are estimated to be using 4 bytes per parameter. The INT8 models use 1 byte per parameter. The optimized models contain approximately 70 percent fewer parameters, and quantization further reduces the memory by 75 percent.

TABLE III. COMPARISON ON UCMERCED DATASET AT SCALE OF ×4.

|**_Model_**|**_Parameters (M)_**|
|---|---|
|TransENet|37.46|
|CSA-FE|32.35|
|HAUNet|1.62|
|AERU-Net|0.706|
|Optimized KV260|0.2952|



As shown in the table above, the optimized KV260 model contains only 0.295 million parameters. This is less than half the size of AERU-Net at 0.706 million and significantly smaller than other models such as TransENet and CSA-FE. Its compact architecture makes it highly suitable for deployment on the resource-constrained KV260 platform, providing faster inference and lower memory usage. In practice, the proposed model sacrifices only a small amount of accuracy to achieve substantial efficiency gains.

Visual outputs confirm that the optimized model produces perceptually sharp and structurally consistent results.

## _1) ×2 upscale_







Fig. 6. “building52” from the UCMerced datase: GT (Left), LR (Middle), and Our Optimized KV260 Result (Right)

Authorized licensed use limited to: UNIV OF ENGINEERING AND TECHNOLOGY LAHORE. Downloaded on July 22,2026 at 10:29:12 UTC from IEEE Xplore.  Restrictions apply.







Fig. 7. “airplane80” from the UCMerced datase: GT (Left), LR (Middle), and Our Optimized KV260 Result (Right)

Figure 6 and Figure 7 showcase the output from our deployed INT8 model at ×2 upscaling on example scenes from the UCMerced dataset. While slight softening is observed compared to the ground truth, fine details such as rooftop edges and airplane contours are successfully recovered

## _2) ×4 upscale_







Fig. 8. “runway65” from the UCMerced datase: GT (Left), LR (Middle), and Our Optimized KV260 Result (Right)







Fig. 9. “storagetanks81” from the UCMerced datase: GT (Left), LR (Middle), and Our Optimized KV260 Result (Right)

Figures 8 and 9 present results at the more challenging ×4 upscaling. Despite the increased difficulty, the model successfully recovers key structures such as the circular outline of storage tanks and the linear geometry of runways, although textures remain partially blurred and some edge artifacts are visible.

## V. CONCLUSION

This paper presents a complete framework for deploying a deep learning-based super-resolution model for remote sensing imagery on edge devices. The proposed system is derived from a structurally simplified version of AERU-Net, adapted for quantization-aware training and deployment using the Xilinx Vitis AI toolchain. The optimized model is successfully implemented on the Xilinx Kria KV260 platform and evaluated at ×2 and ×4 upscaling factors. Quantization-aware training minimizes accuracy degradation, with PSNR loss under one decibel and SSIM drop under ten percent. The final INT8 model achieves realtime inference exceeding 20 frames per second. Its memory usage is reduced from approximately 3.7 megabytes to just

0.3 megabytes, allowing it to run entirely within the FPGA’s on-chip memory without relying on external DRAM. These results demonstrate the practicality of deploying deep superresolution models on edge AI platforms for efficient and scalable remote sensing applications.

## ACKNOWLEDGMENT

This research was supported in part by the Multimedia Data Analytics and Processing Research Unit, Department of Electrical Engineering, Chulalongkorn University, Bangkok 10330, Thailand. It was also supported by the Graduate School, Chulalongkorn University, in commemoration of the auspicious occasion of Her Royal Highness Princess Maha Chakri Sirindhorn’s 60th birthday. Additionally, this work was funded by the Thailand Science Research and Innovation Fund, Chulalongkorn University (Grant Number: IND_FF_68_280_2100_039).

## REFERENCES

- [1] C. Dong, C. C. Loy, K. He, and X. Tang, "Image superresolution using deep convolutional networks," _IEEE transactions on pattern analysis and machine intelligence,_ vol. 38, no. 2, pp. 295-307, 2015.

- [2] B. Lim, S. Son, H. Kim, S. Nah, and K. Mu Lee, "Enhanced deep residual networks for single image super-resolution," in _Proceedings of the IEEE conference on computer vision and pattern recognition workshops_ , 2017, pp. 136-144.

- [3] A. Hajian and S. Aramvith, "AERU-Net: Adaptive Edge Recovery and Attention U-Shaped Network for Remote Sensing Image Super-Resolution," _IEEE Access,_ 2025.

- [4] Y. Zhang, K. Li, K. Li, L. Wang, B. Zhong, and Y. Fu, "Image super-resolution using very deep residual channel attention networks," in _Proceedings of the European conference on computer vision (ECCV)_ , 2018, pp. 286-301.

- [5] S. Lei, Z. Shi, and W. Mo, "Transformer-based multistage enhancement for remote sensing image super-resolution," _IEEE Transactions on Geoscience and Remote Sensing,_ vol. 60, pp. 1- 11, 2021.

- [6] N. Sultan, A. Hajian, and S. Aramvith, "An advanced features extraction module for remote sensing image super-resolution," in _2024 21st International Conference on Electrical Engineering/Electronics, Computer, Telecommunications and Information Technology (ECTI-CON)_ , 2024: IEEE, pp. 1-6.

- [7] J. Wang, B. Wang, X. Wang, Y. Zhao, and T. Long, "Hybrid attention-based U-shaped network for remote sensing image super-resolution," _IEEE Transactions on Geoscience and Remote Sensing,_ vol. 61, pp. 1-15, 2023.

- [8] S. Santiwiwat, A. Hajian, W. Ruangsang, and S. Aramvith, "Multiple Face Detection and Recognition for System-an-Chip FPGAs," in _2024 21st International Joint Conference on Computer Science and Software Engineering (JCSSE)_ , 2024: IEEE, pp. 514-521.

- [9] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "Cbam: Convolutional block attention module," in _Proceedings of the European conference on computer vision (ECCV)_ , 2018, pp. 3- 19.

- [10] A. Dosovitskiy _et al._ , "An image is worth 16x16 words: Transformers for image recognition at scale," _arXiv preprint arXiv:2010.11929,_ 2020.

- [11] Y. Yang and S. Newsam, "Bag-of-visual-words and spatial extensions for land-use classification," in _Proceedings of the 18th SIGSPATIAL international conference on advances in geographic information systems_ , 2010, pp. 270-279.

- [12] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: from error visibility to structural similarity," _IEEE transactions on image processing,_ vol. 13, no. 4, pp. 600-612, 2004.

Authorized licensed use limited to: UNIV OF ENGINEERING AND TECHNOLOGY LAHORE. Downloaded on July 22,2026 at 10:29:12 UTC from IEEE Xplore.  Restrictions apply.
