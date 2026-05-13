# open_vcv

**Augmentation-Aware Intersect-Union Decomposition**  
Unsupervised Visual Representation Learning

---

## Ý tưởng

Phương pháp học biểu diễn ảnh không giám sát mới, phân tách embedding thành:
- **Intersect vector**: đặc trưng bất biến (invariant) qua các augmentation
- **Unique vector**: đặc trưng riêng biệt (equivariant) của từng phiên bản augment

Điểm mới: trọng số loss được tính **động** dựa trên mức độ tương đồng thực sự giữa các ảnh augment.

→ Xem chi tiết tại [`IDEA.md`](./IDEA.md)

---

## Cấu trúc

```
open_vcv/
├── IDEA.md          # Ý tưởng, phương pháp, kế hoạch paper
├── README.md        # File này
├── data/            # Datasets
├── papers/          # Reference papers (.pdf)
├── src/             # Source code
│   ├── models/      # Model architectures
│   ├── losses/      # Loss functions
│   ├── datasets/    # Data pipelines
│   ├── trainers/    # Training loops
│   └── utils/       # Utilities
├── experiments/     # Experiment configs và results
├── notebooks/       # Analysis notebooks
└── paper/           # LaTeX paper source
```

---

## Datasets có sẵn (`data/`)

- CIFAR-10 / CIFAR-100
- STL-10
- Tiny ImageNet
- ImageNet

---

## Bắt đầu

Đọc `IDEA.md` để nắm rõ phương pháp, sau đó bắt đầu code trong `src/`.
