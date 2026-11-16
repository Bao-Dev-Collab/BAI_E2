import unittest

from src.tensorboard_logger import (
    flush_summary_writer,
    log_epoch_metrics,
    log_test_metrics,
)


class RecordingWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.flushed = False

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def flush(self) -> None:
        self.flushed = True


class TensorBoardLoggerTests(unittest.TestCase):
    def test_log_epoch_metrics_writes_expected_tags(self) -> None:
        writer = RecordingWriter()

        log_epoch_metrics(
            writer,
            epoch=2,
            train_metrics={"loss": 0.8, "accuracy": 0.7},
            train_eval_metrics={"loss": 0.6, "accuracy": 0.8},
            val_metrics={"loss": 0.5, "accuracy": 0.85},
            learning_rate=0.001,
            batch_size=16,
        )

        self.assertEqual(
            writer.scalars,
            [
                ("Loss/train", 0.8, 2),
                ("Loss/train_eval", 0.6, 2),
                ("Loss/validation", 0.5, 2),
                ("Accuracy/train", 0.7, 2),
                ("Accuracy/train_eval", 0.8, 2),
                ("Accuracy/validation", 0.85, 2),
                ("Hyperparameters/learning_rate", 0.001, 2),
                ("Hyperparameters/batch_size", 16, 2),
            ],
        )

    def test_log_test_metrics_and_flush_writer(self) -> None:
        writer = RecordingWriter()

        log_test_metrics(writer, {"loss": 0.3, "accuracy": 0.9}, epoch=4)
        flush_summary_writer(writer)

        self.assertEqual(
            writer.scalars,
            [("Loss/test", 0.3, 4), ("Accuracy/test", 0.9, 4)],
        )
        self.assertTrue(writer.flushed)


if __name__ == "__main__":
    unittest.main()
