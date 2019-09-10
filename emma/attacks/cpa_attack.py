import numpy as np
from celery.utils.log import get_task_logger
from emma.attacks.leakagemodels import LeakageModel
from emma.utils.utils import EMMAException

logger = get_task_logger(__name__)


def cpa_attack_trace_set(trace_set, result, conf):
    if result.guess_sample_score_matrix is None:
        raise EMMAException("Score matrix is not initialized.")

    if not trace_set.windowed:
        logger.warning("Trace set not windowed. Skipping attack.")
        return

    if trace_set.num_traces <= 0:
        logger.warning("Skipping empty trace set.")
        return

    hypotheses = np.empty([256, trace_set.num_traces])

    # 1. Build hypotheses for all 256 possibilities of the key and all traces
    leakage_model = LeakageModel(conf)
    for subkey_guess in range(0, 256):
        for i in range(0, trace_set.num_traces):
            hypotheses[subkey_guess, i] = leakage_model.get_trace_leakages(trace=trace_set.traces[i], subkey_start_index=conf.subkey, key_hypothesis=subkey_guess)

    # 2. Given point j of trace i, calculate the correlation between all hypotheses
    for j in range(0, trace_set.window.size):
        # Get measurements (columns) from all traces
        measurements = np.empty(trace_set.num_traces)
        for i in range(0, trace_set.num_traces):
            measurements[i] = trace_set.traces[i].signal[j]

        # Correlate measurements with 256 hypotheses
        for subkey_guess in range(0, 256):
            # Update correlation
            result.guess_sample_score_matrix.update((subkey_guess, j), hypotheses[subkey_guess, :], measurements)
