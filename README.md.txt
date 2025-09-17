🚀 CIFAR-10 — Knowledge Distillation → Pruning (Colab, Research-Friendly)




🎯 Goal: Apply Knowledge Distillation (KD) from ResNet-18 (Teacher) to MobileNetV3-Small (Student) and then compress with L1 unstructured pruning. Report the ⚖️ accuracy – ⏱️ latency – 💾 size trade-off in a reproducible, single-notebook workflow.

🔍 Summary

🗂️ Data: CIFAR-10 (inputs resized to 224×224, normalized with ImageNet stats)

🧠 Teacher: ResNet-18 (ImageNet-pretrained), briefly fine-tuned on CIFAR-10

🧩 Student: MobileNetV3-Small (trained from scratch)

🧪 KD Loss: KL(T=4.0) + CE (α=0.9) + label smoothing=0.05

✂️ Pruning: L1 unstructured on Conv2d weights (suggested: 20–30%) + short fine-tune

📏 Metrics: Test Accuracy, single-sample GPU Latency (ms), parameter-based Model Size (MB)

💡 Note: Unstructured pruning doesn’t shrink GPU kernel shapes—speedup may be limited. For real speedups consider structured (channel/filter) pruning, INT8 (PTQ/QAT), or graph compilers (ONNX/TensorRT).

🗃️ Project Structure

cifar10-distill-prune-quant-colab/
 ├─ notebooks/
 │   └─ distill_prune_quant_colab.ipynb
 ├─ assets/
 │   ├─ results_gpu_only.csv
 │   └─ metrics_grid.png
 ├─ README.md
 ├─ requirements.txt
 ├─ .gitignore
 ├─ LICENSE
 └─ 
   📓 Notebook: notebooks/distill_prune_quant_colab.ipynb

   📄 Results (CSV): assets/results_gpu_only.csv     


⚙️ Local Setup & Run

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter lab   # or jupyter notebook

➡️ In the notebook, follow: Teacher fine-tune → KD → Pruning + short fine-tune → Evaluation & Plots.
📦 Outputs are saved under assets/: results_gpu_only.csv, metrics_grid.png.

📈 Results

🖼️ Overview: assets/metrics_grid.png

📄 Raw table: assets/results_gpu_only.csv

Example table (numbers vary by hardware/training budget):

| 📎 Variant              | ✅ Accuracy | ⏱️ LatencyMs | 💾 SizeMB |
| ----------------------- | :--------: | -----------: | --------: |
| Teacher-ResNet18 (GPU)  |  0.80–0.92 |        \~2–4 |   \~44–46 |
| Student-KD-Pruned (GPU) |  0.60–0.75 |       \~5–10 |     \~5–7 |


🔬 Method Details


⚙️ Example hyperparams: LR=3e-4, weight_decay=1e-4, batch=128, Teacher epochs≈3, Student KD epochs≈6–8

🧪 Ablations to try:

KD temperature T ∈ {2,4,6}, α ∈ {0.5,0.7,0.9}

Pruning ratio p ∈ {0.2,0.3,0.5} (+ 1–2 epochs recovery)

Augmentations: RandAugment / Mixup

⚠️ Limitations: Unstructured pruning doesn’t reduce kernel dimensions; GPU latency gains are limited.


🧭 Roadmap

🔧 Structured pruning (channel/filter) + retraining & real speed measurements

🧮 INT8: PTQ/QAT; compare CPU vs GPU

🚀 Compilation / acceleration via ONNX Runtime / TensorRT

🌐 Generalization: CIFAR-100 or Tiny-ImageNet


⚖️ License

This project is released under the MIT license (see LICENSE).













