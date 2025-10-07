import logging
import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock

from ghoulfluids.logging import setup_logging, get_logger

class TestLogging(unittest.TestCase):

    def test_setup_logging_stdout(self):
        """Verify that setup_logging configures logging to stdout correctly."""
        setup_logging("DEBUG")
        logger = logging.getLogger()
        self.assertEqual(logger.level, logging.DEBUG)
        self.assertTrue(any(isinstance(h, logging.StreamHandler) for h in logger.handlers))


    def test_setup_logging_file(self):
        """Verify that setup_logging configures file logging correctly."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            log_file_path = tmp_file.name

        try:
            with patch("logging.FileHandler") as mock_file_handler:
                setup_logging("INFO", log_file=log_file_path)
                logger = logging.getLogger()
                self.assertEqual(logger.level, logging.INFO)
                mock_file_handler.assert_called_once_with(log_file_path, mode='a')
        finally:
            os.remove(log_file_path)


    def test_get_logger(self):
        """Verify that get_logger returns a logger and that it logs messages."""
        logger = get_logger("test_logger")
        self.assertIsInstance(logger, logging.Logger)

        # Reset handlers to avoid interference from other tests
        logging.getLogger("test_logger").handlers = []

        with self.assertLogs("test_logger", level="INFO") as cm:
            logger.info("This is an info message.")
            logger.warning("This is a warning message.")
        self.assertEqual(len(cm.output), 2)
        self.assertIn("INFO:test_logger:This is an info message.", cm.output[0])
        self.assertIn("WARNING:test_logger:This is a warning message.", cm.output[1])

    def test_log_to_file(self):
        """Verify that log messages are written to the specified file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as tmp_file:
            log_file_path = tmp_file.name

        try:
            # Set up logging to the temporary file
            setup_logging("INFO", log_file=log_file_path)

            # Get a logger and log a message
            logger = get_logger("file_test_logger")
            test_message = "This is a test message for file logging."
            logger.info(test_message)

            # Shut down the logging system to ensure the file handler flushes
            logging.shutdown()

            # Re-open the file and check the contents
            with open(log_file_path, "r") as f:
                log_contents = f.read()

            self.assertIn(test_message, log_contents)
            self.assertIn("INFO", log_contents)
            self.assertIn("file_test_logger", log_contents)

        finally:
            # Clean up the temporary file
            if os.path.exists(log_file_path):
                os.remove(log_file_path)

if __name__ == "__main__":
    unittest.main()
