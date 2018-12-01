from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import lyx


def main():
    scaler = MinMaxScaler()
    sentFeatureVec = lyx.io.load_pkl("sentFeatureVec")[0]
    # normaalizedVec = list(lyx.common.mp_map(lambda vec: normalize(
    #     vec, norm='l1', axis=1), sentFeatureVec))
    scaler.fit(sentFeatureVec)
    normaalizedVec = scaler.transform(sentFeatureVec)
    lyx.io.save_pkl(normaalizedVec, "normaalizedVec")
    lyx.io.save_pkl(scaler, "scaler")


if __name__ == "__main__":
    main()
