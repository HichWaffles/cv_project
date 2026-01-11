import json
import os
from collections import deque
from functools import wraps
from flask import Flask, jsonify, render_template, request
import webview
import cv2
import numpy as np
import mediapipe as mp
from lib.detection import GestureDetectionEngine
from lib.config import InferenceConfig

gui_dir = os.path.join(os.path.dirname(__file__), '..', 'gui')
if not os.path.exists(gui_dir):
    gui_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui')

server = Flask(__name__, static_folder=gui_dir, template_folder=gui_dir)
server.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

detection_engine = GestureDetectionEngine.get_instance()


def verify_token(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        data = json.loads(request.data)
        token = data.get('token')
        if token == webview.token:
            return function(*args, **kwargs)
        else:
            raise Exception('Authentication error')
    return wrapper


@server.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response


@server.route('/')
def landing():
    return render_template('index.html', token=webview.token)

@server.route('/api/predict', methods=['POST'])
@verify_token
def predict_gesture():
    """Process frame and predict gesture"""
    global detection_engine
    
    if detection_engine is None:
        return jsonify({
            'success': False,
            'error': 'Detection engine not initialized'
        }), 400
    
    try:
        data = json.loads(request.data)
        
        # Get frame data (base64 encoded or file upload)
        if 'frame' not in data:
            return jsonify({
                'success': False,
                'error': 'No frame data provided'
            }), 400
        
        # Decode frame (assuming base64 encoded image)
        import base64
        frame_data = base64.b64decode(data['frame'])
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process frame
        result = detection_engine.process_frame(frame)
        
        if result is None:
            return jsonify({
                'success': True,
                'detected': False,
                'message': 'No hand detected'
            })
        
        return jsonify({
            'success': True,
            'detected': True,
            'gesture': result['gesture'],
            'confidence': result['confidence'],
            'timestamp': result['timestamp']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@server.route('/api/statistics', methods=['POST'])
@verify_token
def get_statistics():
    """Get detection statistics for the specified window"""
    global detection_engine
    
    if detection_engine is None:
        return jsonify({
            'success': False,
            'error': 'Detection engine not initialized'
        }), 400
    
    try:
        data = json.loads(request.data)
        window_ms = data.get('window_ms', InferenceConfig.WINDOW_MS)
        
        stats = detection_engine.get_recent_statistics(window_ms)
        
        return jsonify({
            'success': True,
            'statistics': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@server.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update configuration"""
    global detection_engine
    
    if request.method == 'GET':
        return jsonify({
            'success': True,
            'config': InferenceConfig.to_dict()
        })
    
    elif request.method == 'POST':
        try:
            data = json.loads(request.data)
            
            # Verify token for POST
            if data.get('token') != webview.token:
                raise Exception('Authentication error')
            
            # Update config
            InferenceConfig.update_from_dict(data.get('config', {}))
            
            # Reinitialize detection engine MediaPipe settings if needed
            if detection_engine is not None:
                # Update MediaPipe settings
                mp_hands = mp.solutions.hands
                detection_engine.hand_detector = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=InferenceConfig.MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=InferenceConfig.MIN_TRACKING_CONFIDENCE
                )
            
            return jsonify({
                'success': True,
                'message': 'Configuration updated',
                'config': InferenceConfig.to_dict()
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500


@server.route('/api/clear_history', methods=['POST'])
@verify_token
def clear_history():
    """Clear detection history"""
    global detection_engine
    
    if detection_engine is None:
        return jsonify({
            'success': False,
            'error': 'Detection engine not initialized'
        }), 400
    
    try:
        with detection_engine.lock:
            detection_engine.detection_history.clear()
            detection_engine.last_keypress_time.clear()
        
        return jsonify({
            'success': True,
            'message': 'History cleared'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500