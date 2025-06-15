from datetime import datetime

LOG_DETAIL = False
BACKTESTING = False

def format_timestamp(current_timestamp):
    return datetime.fromtimestamp(current_timestamp).strftime('%Y%m%d-%H%M%S')

def format_msg(msg, current_timestamp):
    if BACKTESTING:
        if current_timestamp:
            return f'[{format_timestamp(current_timestamp)}] {msg}'
        return msg
    else:
        return msg