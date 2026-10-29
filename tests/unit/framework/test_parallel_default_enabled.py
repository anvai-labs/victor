"""TDD tests for enabling parallel pipeline execution by default."""


class TestParallelDefaultEnabled:

    def test_pipeline_defaults_parallel_true(self):
        from victor.framework.vertical_integration import VerticalIntegrationPipeline

        pipeline = VerticalIntegrationPipeline()
        assert (
            pipeline._parallel_enabled is True
        ), "VerticalIntegrationPipeline should default to parallel_enabled=True"

    def test_parallel_can_be_explicitly_disabled(self):
        from victor.framework.vertical_integration import VerticalIntegrationPipeline

        pipeline = VerticalIntegrationPipeline(parallel_enabled=False)
        assert pipeline._parallel_enabled is False

    def test_create_integration_pipeline_defaults_parallel(self):
        from victor.framework.vertical_integration import create_integration_pipeline

        pipeline = create_integration_pipeline()
        assert pipeline._parallel_enabled is True

    def test_create_integration_pipeline_can_disable(self):
        from victor.framework.vertical_integration import create_integration_pipeline

        pipeline = create_integration_pipeline(enable_parallel=False)
        assert pipeline._parallel_enabled is False
