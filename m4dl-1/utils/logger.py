from pathlib import Path

LOGGER = None

def init(log_file: str) -> None:
    global LOGGER

    # Close existing logger if already initialized (for worker processes)
    if LOGGER is not None:
        LOGGER.close()

    # Create parent directory if needed
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Use line buffering (buffering=1) for automatic flush on newlines
    # This is more efficient than flushing every write, but still ensures
    # logs appear immediately after each complete message
    LOGGER = open(log_file, "w", buffering=1)
    
def shutdown() -> None:
    global LOGGER
    if LOGGER is not None:
        # Flush any remaining buffered writes before closing
        LOGGER.flush()
        LOGGER.close()
        LOGGER = None
        
def log(message: str, end='\n') -> None:
    global LOGGER
    if LOGGER is not None:
        LOGGER.write(message + end)
        # Explicit flush to prevent race conditions in parallel mode
        LOGGER.flush()