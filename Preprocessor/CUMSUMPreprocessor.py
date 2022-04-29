from Preprocessor.Preprocessor import Preprocessor


class CUMSUMPreprocessor(Preprocessor):

    def getTEvents(self, df, upper, lower=None):
        if lower is None:
            lower = - upper
        if upper < 0 or lower > 0:
            raise ValueError('upper and lower bounds must be non-negative')
        tEvents, sPos, sNeg = [], 0, 0
        diff = df.diff()
        for i, change in enumerate(diff.values[1:]):
            sPos, sNeg = max(0, sPos + change), min(0, sNeg + change)

            if sNeg < lower:
                sNeg = 0
                tEvents.append(i)

            if sPos > upper:
                sPos = 0
                tEvents.append(i)

        return df.index[tEvents], sPos, sNeg
