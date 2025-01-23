import os
import re
import numpy as np
import datetime
from functools import wraps
from collections import deque
from collections import Counter

DTYPE = np.float32

class Logger:
    def __init__(self, log_file, buffer_size=500):
        self.log_dir = "reinforcement-learning"
        self.log_file = os.path.join(self.log_dir, log_file)
        self.buffer = deque(maxlen=buffer_size)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def log(self, func_name, message, level="INFO"):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] [{level}] [{func_name}] {message}\n"
        self.buffer.append(log_message)
        
        # Buffer dolarsa flush yap
        if len(self.buffer) == self.buffer.maxlen:
            self.flush()
    
    def flush(self):
        if not self.buffer:
            return
            
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.writelines(self.buffer)
        self.buffer.clear()


def load_and_save(filepath, new_data, chunk_size=1000):
    try:
        new_data = new_data.astype(DTYPE)  # Convert to float32
        if os.path.exists(filepath):
            existing_data = np.load(filepath, allow_pickle=True)
            # Save in chunks
            chunks = len(new_data) // chunk_size + 1
            for i in range(chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(new_data))
                chunk = new_data[start_idx:end_idx]
                combined_data = np.concatenate([existing_data, chunk])
                np.save(filepath, combined_data)
                existing_data = combined_data
                del combined_data  # Free memory
        else:
            np.save(filepath, new_data)
        return True
            
    except Exception as e:
        print(f"Kayıt hatası: {str(e)}")
        return False



def log_this(logger):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.log(func.__name__, f"Started with args: {args}, kwargs: {kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.log(func.__name__, "Completed successfully")
                return result
            except Exception as e:
                logger.log(func.__name__, f"Error: {str(e)}", level="ERROR")
                logger.flush()  # Error durumunda hemen flush et
                raise
        return wrapper
    return decorator