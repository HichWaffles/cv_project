"""
Application stub
"""

from lib.detection import GestureDetectionEngine


def initialize():
    detection_engine = GestureDetectionEngine.get_instance()

    print("Initialization complete.")
    return True