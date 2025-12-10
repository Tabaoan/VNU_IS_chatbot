from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# Khởi tạo OCR
ocr = PaddleOCR(lang='vi', use_angle_cls=True)

# Đường dẫn ảnh
image_path = r"C:\Users\tabao\OneDrive\Pictures\Screenshots\Screenshot 2025-12-02 133435.png"

# Chạy OCR
result = ocr.predict(image_path, cls=True)

# In text
for line in result:
    for word in line:
        print(word[1][0])

# ======= Vẽ bounding box thủ công =======
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

for line in result:
    for box, (text, score) in line:
        draw.polygon([tuple(point) for point in box], outline="red", width=2)
        draw.text((box[0][0], box[0][1]-15), text, fill="red")

image.save("ket_qua_ocr.jpg")
print("Đã lưu ảnh kết quả vào ket_qua_ocr.jpg")
