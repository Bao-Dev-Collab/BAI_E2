# Lab 2: Transfer Learning (VGG16 & DenseNet121 on Oxford-IIIT Pet)

## 📖 Tổng quan dự án

Dự án tập trung nghiên cứu và triển khai kỹ thuật **Học chuyển giao (Transfer Learning)** trong bài toán phân loại ảnh đa lớp (37 giống chó và mèo) trên bộ dữ liệu **Oxford-IIIT Pet**. 

Chương trình sử dụng hai kiến trúc mạng nơ-ron tích chập sâu tiêu biểu là **VGG16** và **DenseNet121** (pre-trained trên ImageNet), kết hợp với quy trình kiểm tra trùng lặp dữ liệu, tiền xử lý ảnh, huấn luyện theo từng giai đoạn (Feature Extraction và Fine-tuning) và đánh giá toàn diện hiệu năng mô hình.

---

## 📁 Cấu trúc thư mục

Dự án được tổ chức theo cấu trúc chuẩn hóa nhằm tách biệt rõ ràng giữa mã nguồn, tài liệu báo cáo và kết quả thực nghiệm:

```text
BAI_E2/
├── src/                        # Mã nguồn chính của hệ thống
│   ├── main.py                 # Điểm khởi chạy chính (Entry point) điều phối huấn luyện & đánh giá
│   ├── models.py               # Định nghĩa kiến trúc VGG16, DenseNet121 và các lớp phân loại tùy chỉnh
│   ├── data_loader.py          # Tiền xử lý, tăng cường dữ liệu (Data Augmentation) và tạo DataLoader
│   └── engine.py               # Vòng lặp huấn luyện (Training loop), kiểm thử (Validation) và tính toán metrics
├── docs/                       # Hệ thống tài liệu và báo cáo phân tích chi tiết
│   ├── bao_cao_kiem_tra_du_lieu_va_mo_hinh.md
│   ├── bao_cao_thuyet_trinh_thuc_hanh_2.md
│   ├── huong_dan_kiem_tra_yeu_cau_bai_thuc_hanh_2.md
│   ├── ket_qua_phan_tich_thuc_hanh_2.md
│   └── file tóm tắt chương 3 để làm bài E2.md
├── notebooks/                  # Các script phụ trợ kiểm tra và khám phá dữ liệu
│   └── check_dataset_duplicates.py  # Kiểm tra rò rỉ dữ liệu (Data Leakage) và ảnh trùng lặp
├── outputs/                    # Lưu trữ log, trọng số mô hình và kết quả thực nghiệm
│   └── reduced-experiments/    # Kết quả chạy thực nghiệm trên các tập cấu hình thu gọn
├── bao_cao_hinh_anh/           # Biểu đồ trực quan hóa Loss/Accuracy, Confusion Matrix phục vụ báo cáo
├── .gitignore                  # Cấu hình loại trừ các file tạm, dữ liệu nặng khi đẩy lên Git
├── requirements.txt            # Danh sách các thư viện và phiên bản phụ thuộc
└── README.md                   # Tài liệu hướng dẫn tổng quan dự án
