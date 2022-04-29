import numpy as np


class Strategy:
    def __init__(self, pred, prob, out, price, config):
        self._config = config
        self.pred, self.prob, self.out, self.scoring = pred, prob, out, self._config['strategyScore']
        self.price = price

    def action(self):
        return self.pred*self.prob

    def evaluate(self):
        if not (self.scoring in ['ror', 'accuracy']):
            raise Exception('wrong strategy scoring method.')

        action = self.action()

        if self.scoring == 'ror':
            score = (self.out.loc[action.index].ret*action+1).product()

        else:
            from sklearn.metrics import accuracy_score
            score = accuracy_score(self.out.loc[action.index].bin, np.sign(action))

        return score



