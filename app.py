from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path

# Import code chatbot của bạn
from vnuis_chatbot import chatbot, get_vectordb_stats, store

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__, static_folder=str(BASE_DIR))
CORS(app)
@app.route('/')
def index():
    """Serve chatbot.html trên local"""
    return send_from_directory(BASE_DIR, 'chatbot.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint để chat"""
    try:
        data = request.json
        message = data.get('message', '')
        session_id = data.get('session_id', 'web_session')

        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Gọi chatbot
        response = chatbot.invoke(
            {"message": message},
            config={"configurable": {"session_id": session_id}}
        )

        return jsonify({
            'success': True,
            'response': response
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Trạng thái VectorDB"""
    try:
        stats = get_vectordb_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """Xóa lịch sử hội thoại"""
    try:
        data = request.json
        session_id = data.get('session_id', 'web_session')

        if session_id in store:
            store[session_id].clear()

        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ===========================
# CHẠY LOCALHOST
# ===========================
if __name__ == '__main__':
    # Nếu đang chạy trên Render → Render sẽ set biến môi trường PORT
    port = int(os.environ.get("PORT", 5000))

    print("\n" + "="*60)
    print(f"FLASK SERVER ĐANG KHỞI ĐỘNG TRÊN CỔNG {port}")
    print("="*60)
    print(f" Mở trình duyệt tại: http://127.0.0.1:{port}")
    print("="*60 + "\n")

    app.run(host='0.0.0.0', port=port, debug=True)
