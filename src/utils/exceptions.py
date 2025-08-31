class TrainingError(Exception):
    pass


class ConfigurationError(Exception):
    pass


class ResourceError(Exception):
    pass


class PipelineError(Exception):
    pass


class ModelLoadError(PipelineError):
    pass


class StateTransitionError(PipelineError):
    pass
