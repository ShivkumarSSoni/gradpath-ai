from gradpath.backend.controllers.evaluation_controller import EvaluationController
from gradpath.backend.domain.classifier_kind import ClassifierKind
from gradpath.backend.schemas.evaluation_request import EvaluationRequest

SOME_REPORT = object()
SOME_RECORDS = object()


class FakeUploadValidationService:
    def __init__(self, raises=None):
        self._raises = raises
        self.validated = []

    def validate(self, records):
        self.validated.append(records)
        if self._raises is not None:
            raise self._raises


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


def test_a_failing_validation_is_reported_without_scoring():
    validation = FakeUploadValidationService(raises=ValueError("bad file"))
    report_service = FakeEvaluationReportService()
    controller = EvaluationController(validation, report_service)
    request = EvaluationRequest(classifier_kind=ClassifierKind.DECISION_TREE, records=SOME_RECORDS)

    response = controller.evaluate_uploaded_records(request)

    assert not response.succeeded
    assert "bad file" in response.error_message
    assert report_service.built_with == []


def test_a_failing_scoring_is_reported_as_an_error_response():
    validation = FakeUploadValidationService()
    report_service = FakeEvaluationReportService(raises=RuntimeError("scoring failed"))
    controller = EvaluationController(validation, report_service)
    request = EvaluationRequest(classifier_kind=ClassifierKind.DECISION_TREE, records=SOME_RECORDS)

    response = controller.evaluate_uploaded_records(request)

    assert not response.succeeded
    assert "scoring failed" in response.error_message
