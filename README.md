# Phân tích cảm xúc đánh giá phim IMDB bằng RNN, LSTM và Machine Learning

## 1. Giới thiệu

Dự án xây dựng và so sánh các mô hình phân tích cảm xúc (*Sentiment Analysis*) trên tập dữ liệu đánh giá phim IMDB. Mục tiêu là phân loại nội dung đánh giá thành hai nhóm: **Positive (tích cực)** và **Negative (tiêu cực)**.

Dự án so sánh các phương pháp Machine Learning truyền thống với mạng nơ-ron cơ bản và các mô hình mạng hồi quy, nhằm đánh giá hiệu quả phân loại thông qua các chỉ số Accuracy, Precision, Recall và F1-score.

## 2. Dữ liệu

- **Tên dữ liệu:** IMDB Movie Reviews
- **Số lượng:** 50.000 đánh giá phim.
- **Nhãn:** Positive và Negative.
- **Phân chia:** 25.000 mẫu huấn luyện và 25.000 mẫu kiểm tra.
- **Nguồn dữ liệu:** Bộ dữ liệu IMDB được cung cấp bởi TensorFlow/Keras.

Dữ liệu được tiền xử lý để phù hợp với từng phương pháp. Các mô hình Machine Learning sử dụng đặc trưng TF-IDF, trong khi RNN và LSTM xử lý chuỗi từ đã được mã hóa thành các chỉ số số học.

## 3. Các mô hình được sử dụng

| Mô hình | Phương pháp biểu diễn dữ liệu |
|---|---|
| Logistic Regression | TF-IDF |
| Linear SVM | TF-IDF |
| MLP | TF-IDF |
| Simple RNN | Chuỗi từ được mã hóa |
| LSTM | Chuỗi từ được mã hóa |

Các mô hình được đánh giá trên cùng tập kiểm tra để so sánh khả năng phân loại cảm xúc.

## 4. Kết quả thực nghiệm

| Mô hình | Accuracy | Precision | Recall | F1-score | Thời gian huấn luyện (giây) |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9051 | 0.9023 | 0.9086 | **0.9055** | 19.32 |
| MLP | 0.9028 | 0.8966 | **0.9106** | 0.9035 | 47.53 |
| Linear SVM | 0.9024 | 0.9021 | 0.9026 | 0.9024 | **14.75** |
| LSTM | 0.8408 | 0.8853 | 0.7832 | 0.8311 | 16.57 |
| Simple RNN | 0.8107 | 0.8370 | 0.7717 | 0.8030 | 28.03 |

*Lưu ý: Kết quả phụ thuộc vào cấu hình mô hình, quá trình tiền xử lý và môi trường chạy thực nghiệm.*

## 5. Nhận xét

- **Logistic Regression** đạt F1-score cao nhất (0.9055) và Accuracy đạt 90,51%, cho kết quả tốt nhất trong các mô hình được thử nghiệm.
- **MLP** đạt Recall cao nhất (0.9106), cho thấy khả năng nhận diện các đánh giá thuộc lớp mục tiêu tốt trong cấu hình hiện tại.
- **Linear SVM** có thời gian huấn luyện ngắn nhất trong các mô hình được so sánh.
- **LSTM** đạt kết quả tốt hơn Simple RNN, với Accuracy tăng từ 81,07% lên 84,08% và F1-score tăng từ 0.8030 lên 0.8311.
- Trong thực nghiệm này, các phương pháp sử dụng TF-IDF hoạt động tốt hơn RNN và LSTM. Điều này cho thấy mô hình phức tạp hơn không nhất thiết đem lại kết quả tốt hơn nếu chưa được tối ưu phù hợp.
- Các đường cong huấn luyện và đánh giá của RNN, LSTM có dấu hiệu chênh lệch giữa kết quả huấn luyện và validation. Có thể cần điều chỉnh số epoch, regularization hoặc kiến trúc mô hình để cải thiện khả năng tổng quát hóa.




