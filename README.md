# Dự án huấn luyện mô hình ngôn ngữ lớn

Dự án này sử dụng thư viện Peft và UnSloth để huấn luyện mô hình ngôn ngữ lớn (LLM) sử dụng các thuật toán Self-Training (SFT), Raward và Policy.

## Cấu trúc dự án

* `train`: Thư mục chứa mã nguồn huấn luyện mô hình
* `data`: Thư mục chứa dữ liệu huấn luyện và kiểm thử
* `lora_models`: Thư mục chứa các mô hình đã được huấn luyện

## Thuật toán huấn luyện

Dự án sử dụng ba thuật toán huấn luyện chính:

1. **Supervised Fine-Tuning (SFT)**: Thuật toán này sử dụng mô hình đã được huấn luyện (trong bài này là Llama-3.2-3B) để huấn luyện thêm trên dữ liệu của của mình.
2. **Raward**: Thuật toán này huấn luyện mô hình để dự đoán phần thưởng cho mô hình, với các đầu vào là một cặp chosen và rejected cho mô hình, các phản hồi chosen sẽ tạo ra score cao hơn so với rejected.
3. **Policy**: Thuật toán này áp dụng cơ chế học tăng cường cho sft model, sử dụng reward model để đánh giá phản hồi cho mô hình, sau cập nhật trọng số để khuyến khích mô hình tạo ra các phản hồi có score cao hơn.

## Mô hình

Dự án sử dụng mô hình ngôn ngữ lớn (LLM) để huấn luyện. Mô hình này được xây dựng dựa trên kiến trúc transformer và được huấn luyện trên dữ liệu lớn.

## Thư viện

Dự án sử dụng hai thư viện chính:

1. **Peft**: Thư viện này cung cấp các công cụ để huấn luyện mô hình ngôn ngữ lớn. Với cơ chế chỉ huấn luyện trên adapter của mô hình từ đó giảm số lượng tham số để huấn luyện mô hình.
2. **UnSloth**: Thư viện này cung cấp các công cụ để huấn luyện mô hình sử dụng thuật toán Supervised Fine-Tuning, Với việc sử dụng UnSloth, tốc độ sẽ được cải thiện đáng kể, hơn nữa là dễ sử dụng nhưng hiệu suất tốt vì các tham số đã được tối ưu tốt nhất cho từng mô hình.
