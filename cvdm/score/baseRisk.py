from abc import ABCMeta, abstractmethod


class BaseRisk(object):
    __metaclass__ = ABCMeta
    features = None

    @abstractmethod
    def score(self, row):
        """
        Given a pandas row or dictionary representation,
        calculate the risk score

        Parameters
        ----------
        row : pandas row representation

        Returns
        ----------
        float: the score
        """
        pass

    def get_features(self, row):
        """
        Get the features associated with this score
        """
        return { k:v for k, v in row.items() if k in self.features }
