import cv2
import numpy as np

# Mở camera
cap = cv2.VideoCapture(0)

while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        break

    # Chuyển ảnh sang màu xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Làm mờ ảnh
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sử dụng adaptive thresholding để xử lý ảnh sáng không đều
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Phát hiện cạnh
    edges = cv2.Canny(thresh, 50, 150)

    # Tìm contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Duyệt qua tất cả contours
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Lọc bỏ các contours nhỏ để giảm nhiễu
        if area < 500:
            continue

        # Xấp xỉ đa giác để xác định số lượng đỉnh
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Lấy số lượng đỉnh
        vertices = len(approx)
        
        # Xác định hình dạng dựa trên số lượng đỉnh
        shape = 'unknown'
        if vertices == 3:
            shape = 'tam giac'
        elif vertices == 4:
            # Tính khoảng cách giữa các đỉnh để phân biệt hình vuông, hình chữ nhật và hình thoi
            d1 = np.linalg.norm(approx[0][0] - approx[1][0])
            d2 = np.linalg.norm(approx[1][0] - approx[2][0])
            d3 = np.linalg.norm(approx[2][0] - approx[3][0])
            d4 = np.linalg.norm(approx[3][0] - approx[0][0])
            
            # Kiểm tra các cặp cạnh đối diện có độ dài tương tự
            if abs(d1 - d3) < 0.1 * d1 and abs(d2 - d4) < 0.1 * d2:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h
                if 0.95 <= aspect_ratio <= 1.05:
                    shape = 'hinh vuong'
                else:
                    shape = 'hinh thoi'  # Các cạnh đối diện tương đồng nhưng không phải hình vuông
            else:
                shape = 'hinh chu nhat'
        elif vertices >= 8:
            shape = 'hinh tron'

        # Vẽ hình dạng và tên
        cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
        cv2.putText(frame, shape, (approx[0][0][0], approx[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("Shape Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
