from abc import ABCMeta, abstractmethod


class BaseRisk(object):
    __metaclass__ = ABCMeta
    features = None
    feat_key = None

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

    def get_feature_keys(self):
        """
        Get the keys / column names for the pandas row
        or dictionary representation
        """
        return self.feat_key


    def get_features(self, row):
        """
        Get the features associated with this score
        """
        return { k:v for k, v in row.items() if k in self.features }
