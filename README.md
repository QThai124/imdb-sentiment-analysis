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


## 5. Nhận xét

- **Logistic Regression** đạt F1-score cao nhất (0.9055) và Accuracy đạt 90,51%, cho kết quả tốt nhất trong các mô hình được thử nghiệm.
- **MLP** đạt Recall cao nhất (0.9106), cho thấy khả năng nhận diện các đánh giá thuộc lớp mục tiêu tốt trong cấu hình hiện tại.
- **Linear SVM** có thời gian huấn luyện ngắn nhất trong các mô hình được so sánh.
- **LSTM** đạt kết quả tốt hơn Simple RNN, với Accuracy tăng từ 81,07% lên 84,08% và F1-score tăng từ 0.8030 lên 0.8311.
- Trong thực nghiệm này, các phương pháp sử dụng TF-IDF hoạt động tốt hơn RNN và LSTM. Điều này cho thấy mô hình phức tạp hơn không nhất thiết đem lại kết quả tốt hơn nếu chưa được tối ưu phù hợp.
- Các đường cong huấn luyện và đánh giá của RNN, LSTM có dấu hiệu chênh lệch giữa kết quả huấn luyện và validation. Có thể cần điều chỉnh số epoch, regularization hoặc kiến trúc mô hình để cải thiện khả năng tổng quát hóa.


## 6. Vì sao Logistic Regression tốt hơn LSTM?

Trong thực nghiệm này, Logistic Regression đạt kết quả tốt hơn LSTM trên tập dữ liệu IMDB: Accuracy đạt **90,51%** và F1-score đạt **0,9055**, trong khi LSTM lần lượt đạt **84,08%** và **0,8311**. Sự khác biệt này có thể được giải thích bởi một số nguyên nhân:

- **TF-IDF phù hợp với bài toán phân loại đánh giá phim:** Phương pháp TF-IDF biểu diễn văn bản dựa trên mức độ quan trọng của các từ và cụm từ. Trong đánh giá cảm xúc, những từ như *excellent*, *terrible*, *amazing* hay *boring* thường mang thông tin hữu ích để xác định cảm xúc. Logistic Regression có thể khai thác hiệu quả những đặc trưng này mà không cần học các biểu diễn văn bản phức tạp.

- **LSTM cần được huấn luyện và tối ưu phù hợp:** Mặc dù LSTM có khả năng học các quan hệ tuần tự và ngữ cảnh trong văn bản, mô hình có nhiều tham số cần tối ưu hơn Logistic Regression. Hiệu quả của LSTM phụ thuộc vào các yếu tố như kiến trúc mạng, kích thước embedding, độ dài chuỗi đầu vào, regularization và siêu tham số huấn luyện.

- **Khả năng tổng quát hóa:** Kết quả huấn luyện và validation của LSTM có sự chênh lệch ở một số epoch, cho thấy mô hình có dấu hiệu overfitting. Điều này có thể khiến LSTM học tốt các mẫu trong tập huấn luyện nhưng chưa khái quát tốt trên dữ liệu chưa thấy.

- **Độ phức tạp không đồng nghĩa với hiệu quả cao hơn:** Với dữ liệu IMDB và cấu hình được sử dụng trong dự án, các đặc trưng TF-IDF kết hợp với Logistic Regression đã cung cấp đủ thông tin để phân loại cảm xúc hiệu quả. Vì vậy, mô hình đơn giản hơn vẫn có thể vượt trội so với mô hình học sâu khi dữ liệu, cách biểu diễn và quá trình tối ưu chưa phát huy hết lợi thế của mô hình phức tạp.

**Kết luận:** Kết quả thực nghiệm cho thấy Logistic Regression với TF-IDF là lựa chọn hiệu quả hơn LSTM trong cấu hình hiện tại, cả về chất lượng phân loại lẫn chi phí huấn luyện. Tuy nhiên, điều này không có nghĩa Logistic Regression luôn tốt hơn LSTM; việc điều chỉnh kiến trúc và siêu tham số, cũng như cải thiện quá trình tiền xử lý, có thể giúp LSTM đạt kết quả tốt hơn.


