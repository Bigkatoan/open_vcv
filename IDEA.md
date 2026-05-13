# Augmentation-Aware Intersect-Union Decomposition
## Unsupervised Visual Representation Learning

> **Project:** open_vcv  
> **Status:** Research Phase  
> **Last updated:** May 2026

---

## 1. Ý tưởng cốt lõi (Core Idea)

Học cách **phân tách biểu diễn ảnh** thành hai phần có ý nghĩa:

- **Intersect vector** (`v_inter`): Thông tin **bất biến** (invariant) — những đặc trưng không thay đổi dù ảnh bị augment theo cách nào.
- **Unique vector** (`v_unique`): Thông tin **riêng biệt** (equivariant) — những đặc trưng phản ánh sự khác biệt do augmentation tạo ra.

**Điểm sáng tạo chính:** Trọng số loss được tính **động** dựa trên mức độ tương đồng thực sự giữa các ảnh augment — không phải trọng số cố định.

---

## 2. Phương pháp (Method)

### 2.1 Cấu trúc dữ liệu

```
1 core image → augment thành q phiên bản
  ├─ Aug₁: Conservative  (minimal: flip, crop nhỏ)
  ├─ Aug₂: Moderate      (vừa: color jitter, rotation)
  └─ Aug₃: Aggressive    (mạnh: cutout, solarize)

+ k auxiliary images  →  negative samples (không có nhãn)
```

### 2.2 Kiến trúc mô hình

```
Input Image
    │
    ▼
[Backbone Encoder]  (CNN / ViT)
    │
    ├──► [Image Feature Branch]  →  image_feat  (dùng tính aug similarity)
    │
    ├──► [Intersect Extractor]   →  v_inter     (shared features)
    │
    └──► [Unique Extractor]      →  v_unique    (per-augmentation features)
```

### 2.3 Hàm Loss

**Tổng loss:**

$$\mathcal{L} = \lambda_1 \mathcal{L}_{\text{inter}} + \lambda_2 \mathcal{L}_{\text{unique}} + \lambda_3 \mathcal{L}_{\text{neg}} + \lambda_4 \mathcal{L}_{\text{aux}}$$

#### `L_inter` — Kéo các intersect vectors lại gần nhau

$$\mathcal{L}_{\text{inter}} = \sum_{i,j} w_{ij} \cdot \left(1 - \cos(v_{\text{inter}}^i, v_{\text{inter}}^j)\right)$$

$$w_{ij} = \cos(\text{image\_feat}^i, \text{image\_feat}^j) \quad \text{(aug similarity)}$$

> Aug tương tự → $w_{ij}$ lớn → kéo mạnh hơn

#### `L_unique` — Đẩy các unique vectors ra xa nhau

$$\mathcal{L}_{\text{unique}} = \sum_{i,j} (1 - w_{ij}) \cdot \max(0,\ m_{ij} - \cos(v_{\text{unique}}^i, v_{\text{unique}}^j))$$

$$m_{ij} = 0.5 \times w_{ij} \quad \text{(adaptive margin)}$$

> Aug khác nhau → $(1 - w_{ij})$ lớn → đẩy mạnh hơn

#### `L_neg` — Đẩy core vectors ra xa auxiliary vectors

$$\mathcal{L}_{\text{neg}} = \sum_{i,k} \max(0,\ \margin - \cos(v_{\text{inter}}^i, v_{\text{aux}}^k))$$

#### `L_aux` — Ngăn auxiliary vectors sụp đổ (collapse prevention)

$$\mathcal{L}_{\text{aux}} = -\sum_{k \neq l} \cos(v_{\text{aux}}^k, v_{\text{aux}}^l)$$

---

## 3. So sánh với các phương pháp hiện tại

| Aspect | SimCLR | MoCo | DINO | **Phương pháp này** |
|--------|--------|------|------|---------------------|
| Labels cần? | Không | Không | Không | **Không** |
| Aug weighting | Đều nhau | Đều nhau | Đều nhau | **Dynamic (theo similarity)** |
| Phân tách inter/unique | Không | Không | Không | **✓ Có** |
| Interpretable | Thấp | Thấp | Trung bình | **Cao** |
| Compression | Không | Không | Không | **60–70% sparsity** |

---

## 4. Novelty (Điểm mới đóng góp)

1. **Augmentation-aware weighting** — Trọng số loss được tính động theo cosine similarity giữa image features, không phải heuristic cố định.
2. **Explicit decomposition** — Học đồng thời cả intersect (invariant) lẫn unique (equivariant) features trong một khung thống nhất.
3. **Adaptive margin** — Margin trong L_unique thay đổi theo mức độ aug, tránh push quá mức với các aug tương tự nhau.
4. **Interpretability** — Có thể phân tích được tại sao mô hình ra quyết định gì thông qua việc kiểm tra v_inter và v_unique.

---

## 5. Kế hoạch thực nghiệm (Experiments)

### 5.1 Datasets
- [ ] CIFAR-10 / CIFAR-100 (baseline nhanh)
- [ ] STL-10 (unlabeled data, phù hợp unsupervised)
- [ ] Tiny ImageNet
- [ ] ImageNet-1K (scale up)

### 5.2 Baselines so sánh
- [ ] SimCLR
- [ ] MoCo v2 / v3
- [ ] BYOL
- [ ] DINO / DINOv2
- [ ] Barlow Twins
- [ ] VICReg

### 5.3 Evaluation metrics
- **Decomposition quality:**
  - Intersect consistency: $\text{cosine\_sim}(v_{\text{inter}}^i, v_{\text{inter}}^j)$ across augmentations (target > 0.85)
  - Unique divergence: độ phân kỳ giữa các $v_{\text{unique}}$ (target > 0.70)
  - Orthogonality: $\langle v_{\text{inter}}, v_{\text{unique}} \rangle \approx 0$ (target < 0.10)

- **Downstream task:**
  - Linear probe accuracy trên labeled test set
  - Few-shot classification (1-shot, 5-shot)
  - Transfer learning (CIFAR → STL-10)

- **Compression:**
  - Model sparsity (target 60–70%)
  - Inference speed vs baseline

### 5.4 Ablation studies
- [ ] Không có aug weighting (dùng weight cố định)
- [ ] Không có L_neg
- [ ] Không có L_aux
- [ ] q = 2 vs q = 3 vs q = 4
- [ ] k = 1 vs k = 2 vs k = 4
- [ ] CNN backbone vs ViT backbone

---

## 6. Cấu trúc paper (Paper Outline)

```
1. Introduction
   - Motivation: unsupervised learning at scale
   - Problem: existing methods treat all augmentations equally
   - Contribution: augmentation-aware decomposition

2. Related Work
   - Contrastive learning (SimCLR, MoCo, BYOL)
   - Self-supervised with equivariance (VICReg, Barlow Twins)
   - Representation decomposition (DINO, iBOT)

3. Method
   3.1 Problem formulation
   3.2 Data pipeline (q augmentations + k negatives)
   3.3 Model architecture
   3.4 Loss functions (L_inter, L_unique, L_neg, L_aux)
   3.5 Augmentation-aware weighting mechanism

4. Experiments
   4.1 Experimental setup
   4.2 Main results (linear probe)
   4.3 Decomposition quality analysis
   4.4 Ablation studies
   4.5 Visualization

5. Analysis & Discussion
   - Why does aug-aware weighting help?
   - What does v_inter capture vs v_unique?
   - Failure cases

6. Conclusion
```

---

## 7. Hyperparameter mặc định (gợi ý bắt đầu)

| Param | Giá trị | Ghi chú |
|-------|---------|---------|
| `q` (số augmentations) | 3 | conservative / moderate / aggressive |
| `k` (số negatives) | 2 | auxiliary images |
| `batch_size` | 64–256 | tùy GPU |
| `lr` | 1e-3 | AdamW, cosine decay |
| `epochs` | 100–200 | 3-phase training |
| `λ_inter` | 1.0 | intersect loss weight |
| `λ_unique` | 0.5 | unique loss weight |
| `λ_neg` | 0.3 | negative loss weight |
| `λ_aux` | 0.1 | aux unique loss weight |
| `margin` | 0.5 | base margin cho L_neg |
| `dim_inter` | 128 | intersect feature dim |
| `dim_unique` | 64 | unique feature dim per head |

---

## 8. Kết quả kỳ vọng

| Phase | Epoch | L_total | Intersect Sim | Unique Div |
|-------|-------|---------|--------------|------------|
| Init | 10 | 0.8–1.2 | 0.35–0.45 | 0.15–0.25 |
| Decomposition | 50 | 0.4–0.6 | 0.75–0.85 | 0.55–0.70 |
| Refinement | 100 | 0.25–0.40 | **0.85–0.95** | **0.70–0.85** |

---

## 9. Cấu trúc thư mục dự án

```
open_vcv/
├── IDEA.md               ← File này (ý tưởng + kế hoạch)
├── data/                 ← Datasets (CIFAR, STL-10, ImageNet...)
├── papers/               ← Reference papers (.pdf)
├── src/                  ← Code của bạn (tự viết)
│   ├── models/           ← Model architecture
│   ├── losses/           ← Loss functions
│   ├── datasets/         ← Data pipeline
│   ├── trainers/         ← Training loops
│   └── utils/            ← Utilities
├── experiments/          ← Config files và kết quả experiment
├── notebooks/            ← Jupyter notebooks cho phân tích
└── paper/                ← LaTeX source cho paper
```

---

*"The augmentation-aware weighting is the key novelty — it lets the model learn which visual differences actually matter."*
