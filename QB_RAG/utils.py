from collections import Counter
import logging
import typing as t

logger = logging.getLogger(__name__)


class Ensember:
    """
    Combine multiple llm outputs for same input (n>1) to a single output
    """

    def from_discrete(self, inputs: list[list[t.Dict]], attribute: str):
        """
        Simple majority voting for binary values, ie [0,0,1] -> 0
        inputs: list of list of dicts each containing verdict for a single input
        """

        if not isinstance(inputs, list):
            inputs = [inputs]

        if not all(len(item) == len(inputs[0]) for item in inputs):
            logger.warning("All inputs must have the same length")
            return inputs[0]

        if not all(attribute in item for input in inputs for item in input):
            logger.warning(f"All inputs must have {attribute} attribute")
            return inputs[0]

        if len(inputs) == 1:
            return inputs[0]

        verdict_agg = []
        for i in range(len(inputs[0])):
            item = inputs[0][i]
            verdicts = [inputs[k][i][attribute] for k in range(len(inputs))]
            verdict_counts = dict(Counter(verdicts).most_common())
            item[attribute] = list(verdict_counts.keys())[0]
            verdict_agg.append(item)

        return verdict_agg


ensembler = Ensember()
