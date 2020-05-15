from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.algorithms.algorithm import BaseAlgorithm


n_neighbors = 3
random_state = 0

class KnnAlgorithm(BaseAlgorithm):
    def __init__(self, data_path, unique_values_per_columns, existing_parameters):
        self.polynomial_features = None
        self.existing_parameters = existing_parameters
        super().__init__(data_path, unique_values_per_columns)

    """ Implementation of abstract method """

    def initialize_model_and_scaler(self):
        self.scaler = StandardScaler()
        self.model = KNeighborsClassifier(n_neighbors=self.existing_parameters['n_neighbors'],
                                          weights=self.existing_parameters['weights'],
                                          p=self.existing_parameters['p'])

        # Reduce dimension to 2 with PCA
        # R2 0.49
        #self.pca = make_pipeline(StandardScaler(),
        #                    PCA(n_components=2, random_state=random_state))

        # Reduce dimension to 2 with LinearDiscriminantAnalysis
        # R2 0.72
        #self.lda = make_pipeline(StandardScaler(),
        #                    LinearDiscriminantAnalysis(n_components=2))

        # Reduce dimension to 2 with NeighborhoodComponentAnalysis
        #self.nca = make_pipeline(StandardScaler(),
        #                    NeighborhoodComponentsAnalysis(n_components=2,
        #                                                   random_state=random_state))

    " " " Override " " "
    """"
    def fit(self):
        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

        #self.nca.fit(self.x_train, self.y_train)

        #self.model.fit(self.nca.transform(self.x_train), self.y_train)
        self.model.fit(self.x_train, self.y_train)
    """

    " " " Override " " "
    """
    def r2(self):
        y_predict = self.model.predict(self.nca.transform(self.x_test))
        score = metrics.r2_score(self.y_test, y_predict)
        print("R2 (test):", score)
    """