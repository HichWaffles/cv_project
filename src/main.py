import logging
from contextlib import redirect_stdout
from io import StringIO
from app import initialize

from server import server

import webview

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    stream = StringIO()
    initialize()
    
    with redirect_stdout(stream):
        window = webview.create_window('ASL keyboard', server)
        webview.start()