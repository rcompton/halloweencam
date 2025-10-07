import logging
import sys
import unittest
from unittest.mock import patch, MagicMock

from ghoulfluids.logging import setup_logging, get_logger


class TestLogging(unittest.TestCase):
    @patch("logging.StreamHandler")
    @patch("logging.FileHandler")
    @patch("logging.getLogger")
    def test_setup_logging(
        self,
        mock_get_logger: MagicMock,
        mock_file_handler: MagicMock,
        mock_stream_handler: MagicMock,
    ):
        """Verify that setup_logging configures file and stream handlers correctly."""
        mock_root_logger = MagicMock()
        mock_get_logger.return_value = mock_root_logger

        # --- Test logging to stdout ---
        mock_root_logger.handlers = []
        setup_logging("DEBUG")

        mock_get_logger.assert_called_once_with()
        mock_root_logger.setLevel.assert_called_once_with(logging.DEBUG)
        mock_stream_handler.assert_called_once_with(sys.stdout)
        mock_file_handler.assert_not_called()
        mock_root_logger.addHandler.assert_called_once_with(
            mock_stream_handler.return_value
        )
        mock_stream_handler.return_value.setFormatter.assert_called_once()

        # --- Reset mocks ---
        mock_get_logger.reset_mock()
        mock_file_handler.reset_mock()
        mock_stream_handler.reset_mock()
        mock_root_logger.reset_mock()

        # --- Test logging to a file ---
        mock_root_logger.handlers = [MagicMock()]
        log_file = "test.log"
        setup_logging("INFO", log_file=log_file)

        mock_get_logger.assert_called_once_with()
        mock_root_logger.setLevel.assert_called_once_with(logging.INFO)
        mock_file_handler.assert_called_once_with(log_file)
        mock_stream_handler.assert_not_called()
        mock_root_logger.removeHandler.assert_called_once()
        mock_root_logger.addHandler.assert_called_once_with(
            mock_file_handler.return_value
        )
        mock_file_handler.return_value.setFormatter.assert_called_once()

    def test_get_logger(self):
        """Verify that get_logger returns a logger and that it logs messages."""
        logger = get_logger("test_logger")
        self.assertIsInstance(logger, logging.Logger)

        with self.assertLogs("test_logger", level="INFO") as cm:
            logger.info("This is an info message.")
            logger.warning("This is a warning message.")
        self.assertEqual(len(cm.output), 2)
        self.assertIn("INFO:test_logger:This is an info message.", cm.output[0])
        self.assertIn("WARNING:test_logger:This is a warning message.", cm.output[1])


if __name__ == "__main__":
    unittest.main()
