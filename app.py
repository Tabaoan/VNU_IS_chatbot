from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path

# Import code chatbot của bạn
from vnuis_chatbot import chatbot, get_vectordb_stats, store

BASE_DIR = Path(__file__).resolve().parent

# -------------------------------
# CẤU HÌNH STATIC ĐÚNG CHUẨN RENDER
# -------------------------------
app = Flask(
    __name__,
    static_folder=str(BASE_DIR),      # Cho phép phục vụ file tĩnh trong thư mục hiện tại
    static_url_path=''                # Để URL /abc.png hoạt động
)

CORS(app)

# ============================================
# ROUTE HIỂN THỊ GIAO DIỆN
# ============================================
@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'chatbot.html')

# ============================================
# API CHAT
# ============================================
@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id', 'web_session')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        response = chatbot.invoke(
            {"message": message},
            config={"configurable": {"session_id": session_id}}
        )

        return jsonify({'success': True, 'response': response})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# API STATUS
# ============================================
@app.route('/api/status', methods=['GET'])
def status():
    try:
        stats = get_vectordb_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ============================================
# API CLEAR HISTORY
# ============================================
@app.route('/api/clear', methods=['POST'])
def clear_history():
    try:
        data = request.json
        session_id = data.get('session_id', 'web_session')

        if session_id in store:
            store[session_id].clear()

        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# CHẠY LOCAL + RENDER
# ============================================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))

    print("\n" + "="*60)
    print(f"FLASK SERVER ĐANG KHỞI ĐỘNG TRÊN CỔNG {port}")
    print("="*60)
    print(f" Mở trình duyệt tại: http://127.0.0.1:{port}")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=port, debug=True)
