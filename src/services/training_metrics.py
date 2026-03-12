"""
Training Metrics Reporter

Pushes training metrics to Prometheus Pushgateway for real-time
monitoring in Grafana. All push operations are fail-safe — training
never breaks due to a metrics push failure.
"""

import os
import logging
from typing import Dict, Optional

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)

PUSHGATEWAY_URL = os.getenv("PUSHGATEWAY_URL", "pushgateway:9091")


class TrainingMetricsReporter:
    """Pushes per-epoch training metrics to Prometheus Pushgateway."""

    def __init__(self, run_id: str, model_name: str = "bloodcells-classifier"):
        self.run_id = run_id
        self.model_name = model_name
        self.registry = CollectorRegistry()
        self._grouping_key = {"run_id": run_id}

        labels = ["run_id", "model_name"]

        self.epoch_gauge = Gauge(
            "bloodcell_training_epoch", "Current training epoch",
            labels, registry=self.registry,
        )
        self.loss_gauge = Gauge(
            "bloodcell_training_loss", "Training loss",
            labels, registry=self.registry,
        )
        self.train_acc_gauge = Gauge(
            "bloodcell_training_accuracy", "Training accuracy",
            labels, registry=self.registry,
        )
        self.val_acc_gauge = Gauge(
            "bloodcell_validation_accuracy", "Validation accuracy",
            labels, registry=self.registry,
        )
        self.best_val_acc_gauge = Gauge(
            "bloodcell_training_best_val_acc", "Best validation accuracy so far",
            labels, registry=self.registry,
        )
        self.status_gauge = Gauge(
            "bloodcell_training_status", "Training status (1=running, 0=idle)",
            labels, registry=self.registry,
        )
        self.total_epochs_gauge = Gauge(
            "bloodcell_training_total_epochs", "Total epochs planned",
            labels, registry=self.registry,
        )

        # Final metrics (pushed once at end)
        self.test_acc_gauge = Gauge(
            "bloodcell_training_test_accuracy", "Test accuracy after training",
            labels, registry=self.registry,
        )
        self.macro_f1_gauge = Gauge(
            "bloodcell_training_macro_f1", "Macro F1 after training",
            labels, registry=self.registry,
        )
        self.class_f1_gauge = Gauge(
            "bloodcell_training_class_f1", "Per-class F1 score",
            ["run_id", "model_name", "class_name"], registry=self.registry,
        )

        self._labels = {"run_id": run_id, "model_name": model_name}

    def _push(self) -> None:
        """Push metrics to gateway. Never raises."""
        try:
            push_to_gateway(
                PUSHGATEWAY_URL, job="training",
                registry=self.registry, grouping_key=self._grouping_key,
            )
        except Exception:
            logger.warning("Failed to push metrics to Pushgateway", exc_info=True)

    def report_start(self, total_epochs: int) -> None:
        """Signal that training has started."""
        self.status_gauge.labels(**self._labels).set(1)
        self.total_epochs_gauge.labels(**self._labels).set(total_epochs)
        self.epoch_gauge.labels(**self._labels).set(0)
        self._push()
        logger.info("Training metrics: start reported (epochs=%d)", total_epochs)

    def report_epoch(
        self, epoch: int, train_loss: float, train_acc: float,
        val_acc: float, best_val_acc: float,
    ) -> None:
        """Push metrics for a completed epoch."""
        self.epoch_gauge.labels(**self._labels).set(epoch)
        self.loss_gauge.labels(**self._labels).set(train_loss)
        self.train_acc_gauge.labels(**self._labels).set(train_acc)
        self.val_acc_gauge.labels(**self._labels).set(val_acc)
        self.best_val_acc_gauge.labels(**self._labels).set(best_val_acc)
        self._push()

    def report_end(
        self,
        test_accuracy: Optional[float] = None,
        macro_f1: Optional[float] = None,
        per_class_f1: Optional[Dict[str, float]] = None,
    ) -> None:
        """Signal that training has ended. Push final quality metrics."""
        self.status_gauge.labels(**self._labels).set(0)

        if test_accuracy is not None:
            self.test_acc_gauge.labels(**self._labels).set(test_accuracy)
        if macro_f1 is not None:
            self.macro_f1_gauge.labels(**self._labels).set(macro_f1)
        if per_class_f1:
            for class_name, f1 in per_class_f1.items():
                self.class_f1_gauge.labels(
                    run_id=self.run_id, model_name=self.model_name,
                    class_name=class_name,
                ).set(f1)

        self._push()
        logger.info("Training metrics: end reported")
