# Bài Thực Hành E2: Transfer Learning với VGG16 & DenseNet121 trên tập dữ liệu Oxford Pet

Dự án áp dụng kỹ thuật Transfer Learning sử dụng hai kiến trúc mạng nơ-ron tích chập sâu **VGG16** và **DenseNet121** để phân loại hình ảnh trên bộ dữ liệu **Oxford-IIIT Pet**.

## 📁 Cấu trúc thư mục

* `src/`: Mã nguồn chính của mô hình (`main.py`, `models.py`, `data_loader.py`, `engine.py`).
* `docs/`: Tài liệu hướng dẫn, báo cáo kiểm tra dữ liệu và phân tích kết quả thực nghiệm.
* `notebooks/`: Các script kiểm tra dữ liệu trùng lặp và thử nghiệm phụ trợ.
* `outputs/`: Lưu trữ kết quả huấn luyện và các biểu đồ đánh giá (`reduced-experiments`).
* `bao_cao_hinh_anh/`: Hình ảnh trực quan hóa phục vụ báo cáo.
* `requirements.txt`: Danh sách các thư viện Python cần thiết.

## ⚙️ Cài đặt môi trường

Cài đặt các thư viện phụ thuộc bằng lệnh:

```bash
pip install -r requirements.txt
