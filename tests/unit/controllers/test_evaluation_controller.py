from gradpath.backend.controllers.evaluation_controller import (
    GENERIC_SCORING_FAILURE_MESSAGE,
    EvaluationController,
)
from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.exceptions.invalid_records_error import InvalidRecordsError
from gradpath.backend.schemas.evaluation_request import EvaluationRequest

SOME_REPORT = object()
SOME_RECORDS = object()


class FakeUploadValidationService:
    def __init__(self, raises=None, warnings=None, cleaned_records=None):
        self._raises = raises
        self._warnings = warnings or []
        self._cleaned_records = cleaned_records
        self.validated = []

    def validate(self, records):
        self.validated.append(records)
        if self._raises is not None:
            raise self._raises
        return self._cleaned_records if self._cleaned_records is not None else records, (
            self._warnings
        )


class FakeEvaluationReportService:
    def __init__(self, raises=None):
        self._raises = raises
        self.built_with = []

    def build_report(self, classifier_kind, records):
        if self._raises is not None:
            raise self._raises
        self.built_with.append((classifier_kind, records))
        return SOME_REPORT


def test_a_valid_upload_returns_a_successful_response():
    validation = FakeUploadValidationService()
    report_service = FakeEvaluationReportService()
    controller = EvaluationController(validation, report_service)
    request = EvaluationRequest(classifier_kind=ClassifierKind.DECISION_TREE, records=SOME_RECORDS)

    response = controller.evaluate_uploaded_records(request)

    assert response.succeeded
    assert response.report is SOME_REPORT
    assert validation.validated == [SOME_RECORDS]
    assert response.warnings == []


def test_warnings_from_validation_are_carried_through_on_success():
    validation = FakeUploadValidationService(warnings=["1 row(s) dropped: missing value."])
    report_service = FakeEvaluationReportService()
    controller = EvaluationController(validation, report_service)
    request = EvaluationRequest(classifier_kind=ClassifierKind.DECISION_TREE, records=SOME_RECORDS)

    response = controller.evaluate_uploaded_records(request)

    assert response.succeeded
    assert response.warnings == ["1 row(s) dropped: missing value."]


def test_the_cleaned_records_from_validation_are_what_gets_scored():
    cleaned_records = object()
    validation = FakeUploadValidationService(cleaned_records=cleaned_records)
    report_service = FakeEvaluationReportService()
    controller = EvaluationController(validation, report_service)
    request = EvaluationRequest(classifier_kind=ClassifierKind.DECISION_TREE, records=SOME_RECORDS)

    controller.evaluate_uploaded_records(request)

    assert report_service.built_with == [(ClassifierKind.DECISION_TREE, cleaned_records)]


def test_a_failing_validation_is_reported_without_scoring():
    validation = FakeUploadValidationService(raises=InvalidRecordsError("bad file"))
    report_service = FakeEvaluationReportService()
    controller = EvaluationController(validation, report_service)
    request = EvaluationRequest(classifier_kind=ClassifierKind.DECISION_TREE, records=SOME_RECORDS)

    response = controller.evaluate_uploaded_records(request)

    assert not response.succeeded
    assert response.error_message == "bad file"
    assert "InvalidRecordsError" not in response.error_message
    assert report_service.built_with == []


def test_an_unexpected_scoring_failure_shows_a_generic_message_not_the_raw_error():
    validation = FakeUploadValidationService()
    report_service = FakeEvaluationReportService(raises=RuntimeError("scoring failed"))
    controller = EvaluationController(validation, report_service)
    request = EvaluationRequest(classifier_kind=ClassifierKind.DECISION_TREE, records=SOME_RECORDS)

    response = controller.evaluate_uploaded_records(request)

    assert not response.succeeded
    assert response.error_message == GENERIC_SCORING_FAILURE_MESSAGE
    assert "scoring failed" not in response.error_message
    assert "RuntimeError" not in response.error_message
