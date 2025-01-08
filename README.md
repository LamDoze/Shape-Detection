Shape Detection, một chương trình đơn giản nhưng mạnh mẽ giúp nhận diện các hình dạng hình học (tam giác, hình vuông, hình chữ nhật, hình tròn, và hình thoi) từ camera thời gian thực! 🎉

🛠️ Tính năng nổi bật

  Nhận diện hình dạng: Tự động phát hiện và phân loại các hình như tam giác, hình vuông, hình chữ nhật, hình tròn, và hình thoi.
  Thời gian thực: Xử lý video trực tiếp từ camera để hiển thị kết quả ngay lập tức.
  Giao diện đơn giản: Khung hình video với các hình dạng được vẽ và chú thích trực tiếp.

📦 Công nghệ sử dụng

  Python: Ngôn ngữ lập trình chính.
  OpenCV: Thư viện xử lý ảnh mạnh mẽ.
  NumPy: Hỗ trợ tính toán toán học và xử lý ma trận.

📸 Hướng dẫn sử dụng

  Mở chương trình, camera sẽ tự động bật.
  Đặt các vật thể hình học trước camera (giấy vẽ, mô hình, hoặc vật dụng có hình dạng rõ ràng).
  Chương trình sẽ:
      Vẽ đường viền quanh hình dạng.
      Hiển thị tên hình dạng gần vị trí đỉnh.
  Nhấn q để thoát chương trình.

🌟 Một số điểm nổi bật trong thuật toán

  Chuyển ảnh sang xám: Giảm dữ liệu màu để xử lý nhanh hơn.
  Làm mờ ảnh: Loại bỏ nhiễu bằng bộ lọc Gaussian Blur.
  Ngưỡng thích nghi (Adaptive Thresholding): Giải quyết vấn đề ánh sáng không đồng đều.
  Xác định hình dạng:
      Dựa vào số đỉnh của hình.
      Phân biệt hình vuông và hình chữ nhật qua tỷ lệ cạnh.
      Phân biệt hình thoi và hình vuông qua độ dài cạnh và tỷ lệ khung.

🎨 Demo

![Screenshot 2025-01-08 091400](https://github.com/user-attachments/assets/8cd98071-effd-4cce-97c3-0d916b9e8776)


