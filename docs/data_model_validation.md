# Báo cáo kiểm tra dữ liệu và mô hình

## Dữ liệu

Dataset Oxford-IIIT Pet có 37 lớp, được chia thành train, validation và test:

| Tập | Số ảnh | Ảnh mỗi lớp |
|---|---:|---:|
| Train | 2.941 | 74–80 |
| Validation | 739 | 19–20 |
| Test | 3.669 | 88–100 |

Đã quét ảnh trong cả ba split: không phát hiện file ảnh không đọc được. Script SHA-256 [check_dataset_duplicates.py](./check_dataset_duplicates.py) không tìm thấy ảnh có nội dung byte-identical giữa các split.

**Giới hạn:** phép so sánh SHA-256 không phát hiện ảnh bị đổi định dạng/nén lại hoặc ảnh gần giống về nội dung.

## Mô hình

VGG16 và DenseNet121 đều dùng trọng số pretrained ImageNet và được điều chỉnh đầu ra thành 37 lớp.

- **DenseNet121:** mặc định có 37.925 tham số trainable trên tổng 6.991.781.
- **VGG16:** đã đóng băng các fully-connected layer ẩn pretrained; chỉ classifier mới được train. Số tham số trainable giảm từ 119.697.445 xuống 151.589 trên tổng 134.412.133.
- Tùy chọn `--fine-tune-last-block` vẫn cho phép mở khối convolution cuối để fine-tune.

Việc giảm số tham số trainable ở VGG16 làm giảm dung lượng phần được cập nhật, nhưng không bảo đảm loại bỏ overfitting.

## Huấn luyện và đánh giá

Mã hiện dừng sớm nếu validation loss không cải thiện trong 3 epoch liên tiếp (`--patience 3`), với giới hạn tối đa do `--epochs` đặt. Metric `train_eval` dùng transform cố định giống validation để giúp so sánh train/validation công bằng hơn. Các cấu hình được chọn theo validation loss; test chỉ chạy một lần cho cấu hình đã chọn.

Đã chạy 4 cấu hình: VGG16 và DenseNet121, mỗi mô hình với learning rate 0.01 và 0.001, batch size 16, tối đa 5 epoch. Kết quả được lưu trong `outputs/reduced-experiments/results.csv`.

| Mô hình | LR | Epoch chọn / chạy | Train-eval accuracy | Val loss | Val accuracy | Test accuracy |
|---|---:|---:|---:|---:|---:|---:|
| VGG16 | 0.01 | 4 / 5 | 92,66% | 1,6225 | 87,96% | Chưa đánh giá |
| VGG16 | 0.001 | 4 / 5 | 95,14% | 0,2831 | 91,34% | 88,80% |
| DenseNet121 | 0.01 | 1 / 4 | 88,10% | 0,4766 | 86,60% | Chưa đánh giá |
| DenseNet121 | 0.001 | 4 / 5 | 95,95% | 0,2767 | 91,88% | 91,31% |

Test accuracy chỉ được ghi cho cấu hình validation loss thấp nhất của mỗi kiến trúc. Learning rate 0.001 tốt hơn 0.01 theo validation loss trong các lần chạy này. Batch size được giữ cố định ở 16, nên chưa có kết luận về ảnh hưởng của batch size.

## So sánh hai kiến trúc

Ở các cấu hình được chọn (LR 0.001, batch size 16), DenseNet121 đạt test accuracy 91,31% và test loss 0,3236; VGG16 đạt lần lượt 88,80% và 0,3450. Chênh lệch accuracy là 2,51 điểm phần trăm, nghiêng về DenseNet121. Validation accuracy cũng nghiêng nhẹ về DenseNet121 (91,88% so với 91,34%).

VGG16 có cấu trúc tuần tự, dễ giải thích và làm mô hình đối chiếu rõ ràng; nhược điểm là tổng số tham số lớn (134.412.133 so với 6.991.781 ở DenseNet121). DenseNet121 có các kết nối dày đặc để tái sử dụng đặc trưng và ít tham số hơn; nhược điểm là luồng feature map phức tạp hơn. Chưa đo thời gian/bộ nhớ thực tế riêng từng kiến trúc, nên không kết luận mô hình nào chạy nhanh hơn.

Kết quả chỉ xác nhận DenseNet121 tốt hơn trong lần chạy, split và cấu hình này; không đủ để kết luận ưu thế tổng quát vì chưa có nhiều seed hoặc k-fold.

## Nhận xét về overfitting

Ở checkpoint được chọn, chênh lệch train-eval và validation accuracy khoảng 3,80 điểm phần trăm với VGG16 và 4,07 điểm phần trăm với DenseNet121. Train-eval loss thấp hơn validation loss ở cả hai mô hình. Đây là generalization gap cần theo dõi; số liệu không đủ để khẳng định mô hình hoàn toàn không overfit, nhưng accuracy gap quan sát được không quá lớn.

Các thay đổi trên giúp phát hiện/giảm nguy cơ overfitting nhưng không bảo đảm loại bỏ nó. Thí nghiệm dùng một split cố định và một seed, không chạy k-fold do chi phí huấn luyện cao trên CPU; kết quả có thể thay đổi theo cách chia dữ liệu/seed. Tập test chỉ dùng để đánh giá cấu hình đã chọn bằng validation.
