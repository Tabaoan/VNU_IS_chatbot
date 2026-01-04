# Initialize PaddleOCR instance
from paddleocr import PaddleOCR

# Khai báo đường dẫn đến file ảnh local của bạn
# Ví dụ 1: Ảnh nằm trong cùng thư mục với script (đường dẫn tương đối)
local_image_path = r"C:\Users\tabao\Downloads\gen-n-testocr.jpg"

# Ví dụ 2: Đường dẫn tuyệt đối (trên Windows)
# local_image_path = "C:/Users/tabao/OneDrive/Desktop/NLP/my_local_image.jpg" 

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

# Run OCR inference on the local image 
result = ocr.predict(
    input=local_image_path)  # THAY THẾ BẰNG ĐƯỜNG DẪN CỦA BẠN

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")