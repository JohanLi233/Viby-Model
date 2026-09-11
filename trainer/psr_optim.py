"""Disjoint parameter views: independent state, clipping and update/NaN policy."""


class ParameterView(dict):
    def __init__(self, model, side=False):
        self.owner, self.side = model, side
        super().__init__(self.trainable_parameters())
        self.config = model.config

    def trainable_parameters(self):
        return {
            k: v
            for k, v in self.owner.trainable_parameters().items()
            if (k == "psr") == self.side
        }

    def parameters(self):
        return {
            k: v
            for k, v in self.owner.parameters().items()
            if (k == "psr") == self.side
        }

    def update(self, params):
        self.owner.update(params)
        return self


def split_gradients(grads):
    return {k: v for k, v in grads.items() if k != "psr"}, {
        "psr": grads["psr"]
    } if "psr" in grads else {}
