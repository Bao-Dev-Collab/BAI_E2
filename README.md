# Lab 2 – Transfer Learning

Practice 2 · Chương 3: Convolutional Neural Networks & Transfer Learning

## 1. Giới thiệu

- **Mục tiêu:** Áp dụng kỹ thuật Học chuyển giao (Transfer Learning) từ các mô hình đã tiền huấn luyện trên ImageNet (**VGG16** và **DenseNet121**) để giải quyết bài toán phân loại đa lớp (37 giống chó/mèo) trên bộ dữ liệu thực tế thay vì huấn luyện mạng từ đầu.
- **Nội dung thực hiện:**
  - Kiểm tra tính toàn vẹn của dữ liệu, phát hiện và loại bỏ rò rỉ dữ liệu (Data Leakage) giữa các tập Train / Validation / Test.
  - Chuẩn hóa ảnh (`224×224` theo phân phối ImageNet) và áp dụng Data Augmentation cho tập huấn luyện; sử dụng phép biến đổi cố định cho Validation/Test.
  - Huấn luyện và đánh giá 4 cấu hình thực nghiệm kết hợp 2 kiến trúc (**VGG16**, **DenseNet121**) và 2 mức Learning Rate (`0.01`, `0.001`) với Batch size `16`, tối đa `5 epochs` và Early Stopping (`patience=3`).

Chi tiết yêu cầu bài thực hành: [docs/huong_dan_kiem_tra_yeu_cau_bai_thuc_hanh_2.md](docs/huong_dan_kiem_tra_yeu_cau_bai_thuc_hanh_2.md). **Báo cáo phân tích đầy đủ:** [docs/ket_qua_phan_tich_thuc_hanh_2.md](docs/ket_qua_phan_tich_thuc_hanh_2.md).

**Kết quả chính** (trên tập kiểm thử Oxford-IIIT Pet với cấu hình tốt nhất `LR = 0.001`, `Batch size = 16`):

| Model | Backbone | Best Learning Rate | Test Accuracy | Ghi chú |
| :--- | :--- | :---: | :---: | :--- |
| **Model 1** | VGG16 (Pre-trained ImageNet) | `0.001` | **88.80%** | Đóng băng backbone, huấn luyện classifier mới |
| **Model 2** | DenseNet121 (Pre-trained ImageNet) | `0.001` | **91.31%** | Đạt độ chính xác cao nhất nhờ kết nối dày đặc Dense Blocks |

*(Chi tiết log và số liệu từng cấu hình lưu tại `outputs/reduced-experiments/results.csv`).*

## 2. Dữ liệu và mô hình

- **Dataset:** **Oxford-IIIT Pet** gồm 37 giống chó và mèo (7.349 ảnh), được tổ chức theo chuẩn `ImageFolder`:
  - `train/`: **2.941 ảnh** (tách phân tầng 80% từ split `trainval` gốc).
  - `val/`: **739 ảnh** (tách phân tầng 20% từ split `trainval` gốc).
  - `test/`: **3.669 ảnh** (giữ nguyên split `test` chính thức để đánh giá cuối cùng).
  - Chi tiết kiểm tra dữ liệu: [docs/bao_cao_kiem_tra_du_lieu_va_mo_hinh.md](docs/bao_cao_kiem_tra_du_lieu_va_mo_hinh.md).
- **Kiến trúc 1 (`VGG16`):** Đóng băng các tầng tích chập trích xuất đặc trưng từ ImageNet, thay thế lớp phân loại cuối cho 37 lớp (hỗ trợ tùy chọn `--fine-tune-last-block`).
- **Kiến trúc 2 (`DenseNet121`):** Tận dụng các khối kết nối dày đặc (Dense Blocks) với trọng số ImageNet giúp tối ưu luồng gradient và giảm số lượng tham số.

## 3. Cấu trúc thư mục

```text
BAI_E2/
├── bao_cao_hinh_anh/                     # Biểu đồ trực quan hóa Loss/Accuracy phục vụ báo cáo
├── docs/                                 # Tài liệu kỹ thuật, tóm tắt lý thuyết và báo cáo phân tích
│   ├── bao_cao_kiem_tra_du_lieu_va_mo_hinh.md
│   ├── bao_cao_thuyet_trinh_thuc_hanh_2.md
│   ├── huong_dan_kiem_tra_yeu_cau_bai_thuc_hanh_2.md
│   ├── ket_qua_phan_tich_thuc_hanh_2.md
│   └── file tóm tắt chương 3 để làm bài E2.md
├── notebooks/                            # Script kiểm tra dữ liệu và thử nghiệm phụ trợ
│   └── check_dataset_duplicates.py       # Kiểm tra rò rỉ dữ liệu (Data Leakage) và ảnh trùng lặp
├── outputs/                              # Lưu trữ log TensorBoard, checkpoint và CSV kết quả
│   └── reduced-experiments/              # Kết quả thực nghiệm 4 cấu hình (results.csv)
├── src/                                  # Mã nguồn chính của hệ thống
│   ├── main.py                           # Điểm khởi chạy chính điều phối huấn luyện, tuning & đánh giá
│   ├── models.py                         # Định nghĩa kiến trúc VGG16, DenseNet121 và Classifier
│   ├── data_loader.py                    # Tiền xử lý, Data Augmentation và tạo DataLoader
│   └── engine.py                         # Vòng lặp huấn luyện (Training loop), Validation và tính metrics
├── .gitignore                            # Loại trừ thư mục dataset nặng, .venv và file tạm
├── requirements.txt                      # Danh sách thư viện (PyTorch, torchvision, TensorBoard...)
└── README.md                             # Tài liệu tổng quan và hướng dẫn sử dụng
