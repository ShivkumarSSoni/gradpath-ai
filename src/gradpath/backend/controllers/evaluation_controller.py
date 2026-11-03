"""Entry point for scoring an uploaded file against one saved model."""

import logging

from gradpath.backend.exceptions.invalid_records_error import InvalidRecordsError
from gradpath.backend.schemas.evaluation_request import EvaluationRequest
from gradpath.backend.schemas.evaluation_response import EvaluationResponse
from gradpath.backend.services.evaluation_report_service import EvaluationReportService
from gradpath.backend.services.upload_validation_service import UploadValidationService

# Shown for any failure that is not a recognised InvalidRecordsError, so an
# internal exception class name or raw message is never exposed to the user.
GENERIC_SCORING_FAILURE_MESSAGE = (
    "This file could not be scored due to an unexpected problem. "
    "Please try a different file."
)

logger = logging.getLogger(__name__)


class EvaluationController:
    """Validates an upload, then scores it, never letting either step crash the app."""

    def __init__(
        self,
        upload_validation_service: UploadValidationService,
        evaluation_report_service: EvaluationReportService,
    ) -> None:
        self._upload_validation_service = upload_validation_service
        self._evaluation_report_service = evaluation_report_service

    def evaluate_uploaded_records(self, request: EvaluationRequest) -> EvaluationResponse:
        """Validate and score `request.records`, reporting any failure as text.

        The saved model is only loaded and scored, never retrained, so an
        upload can never change what the comparison table shows.
        """
        try:
            cleaned_records, warnings = self._upload_validation_service.validate(
                request.records
            )
            report = self._evaluation_report_service.build_report(
                request.classifier_kind, cleaned_records
            )
        except InvalidRecordsError as error:
            # The message on this exception is already plain-English and
            # safe to show as-is, with no internal class name attached.
            logger.exception(
                "Evaluating %s on an uploaded file failed validation.",
                request.classifier_kind.display_name,
            )
            return EvaluationResponse(
                classifier_kind=request.classifier_kind, error_message=str(error)
            )
        except Exception:  # noqa: BLE001 - any failure must be reported, not raised
            logger.exception(
                "Evaluating %s on an uploaded file failed.",
                request.classifier_kind.display_name,
            )
            return EvaluationResponse(
                classifier_kind=request.classifier_kind,
                error_message=GENERIC_SCORING_FAILURE_MESSAGE,
            )
        return EvaluationResponse(
            classifier_kind=request.classifier_kind, report=report, warnings=warnings
        )
