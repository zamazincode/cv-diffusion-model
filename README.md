# Low-Light Enhancement Diffusion Model

Android cihazlarda gerçek zamanlıya yakın çalışan düşük ışık iyileştirmesi (low-light enhancement) için optimize edilmiş diffusion modeli.

## ✨ Özellikler

- **4-8 Adımda Inference**: LCM (Latent Consistency Model) ile hızlı denoising
- **Hafif Mimari**: MobileNetV3-style bloklar ile ~3-5M parametre
- **Kuantizasyon Desteği**: FP16/INT8 quantization ile mobil optimize
- **Multi-Platform Export**: ONNX, TFLite formatları
- **Android Optimized**: NNAPI, GPU Delegate desteği

## 📊 Performans Karşılaştırması

| Model Variant | Params | FP32 Size | FP16 Size | INT8 Size | 4-Step Latency* |
|--------------|--------|-----------|-----------|-----------|-----------------|
| Tiny         | ~1M    | 4 MB      | 2 MB      | 1 MB      | ~50ms           |
| Small        | ~3M    | 12 MB     | 6 MB      | 3 MB      | ~100ms          |
| Base         | ~5M    | 20 MB     | 10 MB     | 5 MB      | ~150ms          |
| Large        | ~10M   | 40 MB     | 20 MB     | 10 MB     | ~250ms          |

*Snapdragon 888, 256x256 input, TFLite + GPU Delegate

## 🏗️ Mimari Tasarım

### UNet Mimarisi

```
Input (Low-light + Noisy) ──┐
        │                   │
   ┌────▼────┐              │
   │ Encoder │              │ Skip Connections
   │ (Down)  │              │
   └────┬────┘              │
        │                   │
   ┌────▼────┐              │
   │ Middle  │              │
   │ Block   │              │
   └────┬────┘              │
        │                   │
   ┌────▼────┐              │
   │ Decoder │◄─────────────┘
   │  (Up)   │
   └────┬────┘
        │
   ┌────▼────┐
   │ Output  │
   └─────────┘
```

### Temel Bileşenler

1. **Inverted Residual Block (MobileNetV3-style)**
   - Depthwise Separable Convolution
   - Squeeze-and-Excitation (SE) attention
   - Time conditioning (FiLM)

2. **Linear Attention**
   - O(n) complexity (vs O(n²) standart attention)
   - Mobil cihazlar için kritik optimizasyon

3. **LCM Scheduler**
   - 4-8 adımda denoising
   - Consistency training ile kalite korunumu

## 🚀 Hızlı Başlangıç

### Kurulum

```bash
# Repository klonla
git clone <repo-url>
cd project-1

# Virtual environment (önerilen)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# veya
.\venv\Scripts\activate  # Windows

# Dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basit training
python scripts/train.py --data_dir /path/to/lol_dataset --epochs 100

# Full config
python scripts/train.py \
    --data_dir /path/to/lol_dataset \
    --val_dir /path/to/lol_test \
    --variant small \
    --image_size 256 \
    --batch_size 8 \
    --lr 1e-4 \
    --epochs 100 \
    --use_amp \
    --use_ema \
    --use_wandb
```

### Export

```bash
# ONNX export (FP16)
python scripts/export.py \
    --checkpoint checkpoints/best_model.pt \
    --format onnx \
    --quantize fp16

# TFLite export (INT8)
python scripts/export.py \
    --checkpoint checkpoints/best_model.pt \
    --format tflite \
    --quantize int8

# Android package (tüm gerekli dosyalar)
python scripts/export.py \
    --checkpoint checkpoints/best_model.pt \
    --android
```

### Benchmark

```bash
# ONNX benchmark
python scripts/benchmark.py \
    --model exports/model_fp16.onnx \
    --format onnx \
    --num_runs 100

# TFLite benchmark
python scripts/benchmark.py \
    --model exports/model.tflite \
    --format tflite \
    --threads 4
```

## 📁 Proje Yapısı

```
project-1/
├── src/
│   ├── models/
│   │   ├── efficient_unet.py     # Hafif UNet mimarisi
│   │   ├── lcm_scheduler.py      # LCM denoising scheduler
│   │   └── low_light_diffusion.py # Ana model sınıfı
│   ├── training/
│   │   ├── dataset.py            # Dataset ve augmentation
│   │   └── trainer.py            # Training loop
│   └── export/
│       ├── quantization.py       # FP16/INT8 quantization
│       ├── onnx_export.py        # ONNX export
│       ├── tflite_export.py      # TFLite export
│       └── android_pipeline.py   # Android inference pipeline
├── scripts/
│   ├── train.py                  # Training script
│   ├── export.py                 # Export script
│   └── benchmark.py              # Benchmark script
├── requirements.txt
└── README.md
```

## 🔧 Optimizasyon Stratejileri

### 1. Model Mimarisi

```python
# Kanal yapısı (hafif versiyon)
channels = [32, 64, 128, 256]  # vs standart [64, 128, 256, 512]

# Depthwise separable conv
# Parametre: k²·Cin·Cout → k²·Cin + Cin·Cout
# 3x3 conv, 64→128: 73,728 → 640 + 8,192 = 8,832 params (8x azalma)
```

### 2. Linear Attention

```python
# Standard Attention: O(n²)
attn = softmax(Q @ K.T) @ V

# Linear Attention: O(n)
attn = φ(Q) @ (φ(K).T @ V)  # φ = elu + 1
```

### 3. LCM Training

```python
# 50 DDIM step → 4 LCM step
# Consistency loss ile self-distillation
loss = ||f(x_t, t) - f(x_{t'}, t')||²
```

### 4. Quantization

| Type | Weight | Activation | Speedup | Quality Loss |
|------|--------|------------|---------|--------------|
| FP16 | FP16   | FP16       | ~2x     | Minimal      |
| INT8 Dynamic | INT8 | FP32  | ~2-3x   | Low          |
| INT8 Static  | INT8 | INT8  | ~3-4x   | Medium       |

### 5. Android Optimizations

```java
// TFLite GPU Delegate
GpuDelegate.Options options = new GpuDelegate.Options();
options.setPrecisionLossAllowed(true);  // FP16
options.setInferencePreference(INFERENCE_PREFERENCE_SUSTAINED_SPEED);
GpuDelegate gpuDelegate = new GpuDelegate(options);

// NNAPI Delegate
NnApiDelegate nnApiDelegate = new NnApiDelegate();
```

## 📈 Training Tips

### Dataset Hazırlığı

```
dataset/
├── train/
│   ├── low/      # Low-light images
│   │   ├── 001.png
│   │   └── ...
│   └── high/     # Normal-light images (paired)
│       ├── 001.png
│       └── ...
└── test/
    ├── low/
    └── high/
```

### Önerilen Dataset'ler

- **LOL**: Low-Light dataset (485 train, 15 test)
- **LOL-v2**: Extended LOL (689 train, 100 test)
- **SID**: See-in-the-Dark (RAW images)
- **SICE**: Multi-exposure dataset

### Training Hyperparameters

```yaml
# Önerilen config
variant: small
image_size: 256
batch_size: 8
learning_rate: 1e-4
epochs: 100
loss: mse  # veya huber
use_amp: true
use_ema: true
ema_decay: 0.9999
num_inference_steps: 4
```

## 🤖 Android Integration

### 1. Gradle Dependencies

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

### 2. Model Yükleme

```kotlin
class LowLightEnhancer(context: Context) {
    private val interpreter: Interpreter
    
    init {
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            addDelegate(GpuDelegate())
        }
        
        val model = FileUtil.loadMappedFile(context, "model.tflite")
        interpreter = Interpreter(model, options)
    }
    
    fun enhance(bitmap: Bitmap): Bitmap {
        // Preprocessing
        val input = preprocessImage(bitmap)
        
        // LCM denoising loop (4 steps)
        var latents = generateNoise()
        
        for (timestep in lcmTimesteps) {
            val modelInput = concatenate(latents, input)
            val output = runInference(modelInput, timestep)
            latents = denoisingStep(output, timestep, latents)
        }
        
        // Postprocessing
        return postprocessImage(latents)
    }
}
```

### 3. Kamera Entegrasyonu

```kotlin
class CameraEnhancer : ImageAnalysis.Analyzer {
    private val enhancer = LowLightEnhancer(context)
    
    override fun analyze(image: ImageProxy) {
        val bitmap = image.toBitmap()
        val enhanced = enhancer.enhance(bitmap)
        
        runOnUiThread {
            imageView.setImageBitmap(enhanced)
        }
        
        image.close()
    }
}
```

## ⚡ Performance Tuning

### Model Size vs Quality Trade-off

```
Tiny:  Fast (50ms)  - Acceptable quality - Good for real-time preview
Small: Medium (100ms) - Good quality - Recommended for most cases
Base:  Slower (150ms) - Better quality - For high-quality output
Large: Slow (250ms) - Best quality - For offline processing
```

### Resolution Scaling

```
256x256: Fastest, lower detail
512x512: Balanced
1024x1024: Highest quality, slow
```

### LCM Steps vs Quality

```
4 steps: Fastest, slight artifacts
6 steps: Balanced
8 steps: Best quality, slower
```

## 📚 Referanslar

- [Latent Consistency Models](https://arxiv.org/abs/2310.04378)
- [MobileNetV3](https://arxiv.org/abs/1905.02244)
- [Retinex-Net (Low-Light Enhancement)](https://arxiv.org/abs/1808.04560)
- [Linear Attention](https://arxiv.org/abs/2006.16236)
- [TensorFlow Lite](https://www.tensorflow.org/lite)

## 📄 License

MIT License

## 🤝 Contributing

Pull request'ler memnuniyetle karşılanır!

1. Fork the repo
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

