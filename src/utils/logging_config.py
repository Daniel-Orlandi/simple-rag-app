"""
Logging configuration module for the RAG system.

This module provides a centralized logging setup with configurable levels:
- DEBUG: Detailed information for debugging (document counts, chunks, queries)
- INFO: Execution monitoring (operation start/completion messages)
- WARNING: Warning level events (empty documents, missing optional configs)
- CRITICAL: Errors or higher than warning (failed operations, missing required configs)
"""
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional


# Default logs directory
LOGS_DIR = Path(__file__).parent.parent.parent / "logs"


def generate_log_filename(user_id: str, session_id: Optional[str] = None) -> str:
    """
    Generate a log filename with user ID, session ID, and timestamp.
    
    Format: {user_id}_{session_id}_{dd-mm-yyyy-hh-mm}.log
    
    Args:
        user_id: User identifier.
        session_id: Session identifier. If None, generates a short UUID.
    
    Returns:
        str: Generated filename (not full path).
    
    Example:
        >>> generate_log_filename("user123", "sess456")
        'user123_sess456_05-02-2026-14-30.log'
    """
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    
    timestamp = datetime.now().strftime("%d-%m-%Y-%H-%M")
    return f"{user_id}_{session_id}_{timestamp}.log"


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    logs_dir: Optional[str] = None
) -> str:
    """
    Configure logging for the entire application.
    
    This function sets up logging with a consistent format across all modules.
    Call this once at application startup (e.g., in main.py).
    
    Args:
        level: Logging level. Use logging.DEBUG, logging.INFO, logging.WARNING, 
               or logging.CRITICAL. Default is INFO.
        format_string: Custom format string for log messages. If None, uses default format.
        log_file: Optional explicit path to a log file. If provided, overrides auto-generated filename.
        user_id: User identifier for auto-generated log filename. If provided (without log_file),
                 creates a log file in the logs folder.
        session_id: Session identifier for the log filename. If None, generates a short UUID.
        logs_dir: Directory for log files. Defaults to 'logs' folder in project root.
    
    Returns:
        str: Path to the log file (if created), or empty string if no file logging.
    
    Example:
        >>> import logging
        >>> from src.utils.logging_config import setup_logging
        >>> 
        >>> # Basic setup with INFO level (console only)
        >>> setup_logging()
        >>> 
        >>> # With user tracking - creates logs/user123_sess456_05-02-2026-14-30.log
        >>> log_path = setup_logging(
        ...     level=logging.INFO,
        ...     user_id="user123",
        ...     session_id="sess456"
        ... )
        >>> 
        >>> # Debug mode with auto-generated session ID
        >>> setup_logging(level=logging.DEBUG, user_id="admin")
    
    Logging Levels:
        - DEBUG: Detailed info (document counts, chunk sizes, query previews)
        - INFO: Execution monitoring (operation start/completion)
        - WARNING: Warning events (empty documents, fallback configs)
        - CRITICAL: Errors (failed operations, missing required configs)
    """
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Determine log file path
    log_file_path = ""
    
    if log_file:
        # Explicit log file path provided
        log_file_path = log_file
    elif user_id:
        # Generate log file with user_id and session_id
        logs_directory = Path(logs_dir) if logs_dir else LOGS_DIR
        logs_directory.mkdir(parents=True, exist_ok=True)
        
        filename = generate_log_filename(user_id, session_id)
        log_file_path = str(logs_directory / filename)
    
    # File handler (if log file path is set)
    if log_file_path:
        # Ensure parent directory exists
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        # Log the session info to the file
        root_logger.info(f"Log session started - User: {user_id}, Session: {session_id}")
    
    # Suppress overly verbose third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    
    return log_file_path


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    This is a convenience function equivalent to logging.getLogger(name).
    
    Args:
        name: Logger name, typically __name__ of the calling module.
    
    Returns:
        logging.Logger: Configured logger instance.
    
    Example:
        >>> from src.utils.logging_config import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting process")
    """
    return logging.getLogger(name)
